# Torch
import torch
import torch.nn.functional as F

# I/O
import json
from smart_open import open
from pathlib import Path

# Text manipulation
import re
import spacy

# Tansformers / Models
from transformers import AutoTokenizer
from colbert import ColBERT, maxsim

# Utils
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

# Mistral
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def generate_chunks_and_encoding(doc_id, retrieved, doc_tokenizer, model, corpus, device, batch_size=16):
    chunks = []
    chunk_len = 600
    for ret in retrieved[doc_id]["retrieved_masked"]:
        start = 0
        text = corpus[ret]
        while len(text) - start > chunk_len:
            chunk = text[start: start + chunk_len]
            ends = [x.end() for x in re.finditer(r"\.\s", chunk)]
            if ends:
                if re.search(r"\.$", chunk):
                    final_full_stop = -1
                else:
                    final_full_stop = ends[-1]
            else:
                final_full_stop = -1

            if final_full_stop == -1:
                start += chunk_len
            else:
                chunk = chunk[:final_full_stop-1]
                chunks.append(chunk)
                start = final_full_stop + start
        chunks.append(text[start:])

    with torch.no_grad():
        doc_encodings = []
        num_batches = int(np.ceil(len(chunks) / batch_size))
        for i in range(num_batches):
            batch = ["[D]" + chunk for chunk in chunks[i*batch_size:(i+1)*batch_size]]
            doc = doc_tokenizer(batch, return_tensors="pt", padding="max_length", max_length=256, truncation=True).to(device)
            doc_encoding = model.document_encoder(doc.input_ids, doc.attention_mask).last_hidden_state
            doc_encoding = model.downsampler(doc_encoding)
            mask = torch.tensor(model.mask(doc.input_ids, model.doc_pad_index), device=doc.input_ids.device).unsqueeze(-1).float()
            doc_encoding = doc_encoding * mask
            doc_encoding = F.normalize(doc_encoding, dim=-1)
            doc_encoding = doc_encoding
            doc_encodings.append(doc_encoding)

    doc_encodings = torch.cat(doc_encodings, dim=0)
    return chunks, doc_encodings


def token_recall(target_tokens, pred_tokens):
    correct = 0
    total = 0
    for tok in pred_tokens:
        if tok in target_tokens:
            correct += 1
        total += 1

    return correct, total


def re_identify_one_span(text, retriever, query_tokenizer, doc_encodings, chunks, reader, reader_tokenizer, num_chunks, span_list, entity_type_list, identifier_type_list, re_id_spans, order_id, device="cpu", num_char=200):
    prompt = """Given the following passages:

{retrieved}

Re-identify the fill in the blank (marked with [MASK]) in the text below, only give the value of the [MASK], do not add extra text, give explanations, or output the blank token [MASK]:

{text}

Answer:"""
    mask_indices = []
    index_mask = text.find("[MASK]")
    while index_mask != -1:
        mask_indices.append(index_mask)
        index_mask = text.find("[MASK]", index_mask+6)

    pos = np.random.randint(0, len(mask_indices))
    target = span_list.pop(pos)
    entity_type = entity_type_list.pop(pos)
    identifier_type = identifier_type_list.pop(pos)
    # ner_type = ner_list.pop(pos)
    mask_index_re_id = mask_indices[pos]
    local_text = text[max(0, mask_index_re_id-num_char):min(len(text), mask_index_re_id+num_char+6)]
    intial_space = local_text.find(" ") if mask_index_re_id-num_char > 0 else -1
    final_space = num_char + 6
    if len(text) > mask_index_re_id+num_char+6:
        while local_text.find(" ", final_space+1) != -1:
            final_space = local_text.find(" ", final_space+1)

    start_pos = intial_space+1 + max(0, mask_index_re_id-num_char)
    end_pos = final_space + max(0, mask_index_re_id-num_char) if len(text) > mask_index_re_id+num_char+6 else len(text) - 1
    if final_space > mask_index_re_id - start_pos and mask_index_re_id-start_pos >= 0:
        local_text = local_text[intial_space+1:final_space]
    else:
        print("Problem!")
        print(local_text)
        print(final_space)
        print(mask_index_re_id - start_pos)
        start_pos = max(0, mask_index_re_id-num_char)
        end_pos = max(0, mask_index_re_id-num_char) + len(local_text)

    local_text_ret = local_text
    local_text_read = local_text

    # Retrieve
    local_mask_indices = []
    for j, local_mask_index in enumerate(mask_indices):
        if local_mask_index > end_pos:
            break
        if local_mask_index >= start_pos and j != pos:
            local_mask_indices.append(local_mask_index-start_pos)

    for i, mask_index in enumerate(local_mask_indices):
        local_text_ret = local_text_ret[:mask_index+0] + "[ANON]" + local_text_ret[mask_index+6+0:]

    with torch.no_grad():
        segment = "[Q]" + local_text_ret
        query = query_tokenizer(segment, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
        query.attention_mask[:, :] = 1
        query_encoding = retriever.query_encoder(query["input_ids"], query["attention_mask"]).last_hidden_state
        query_encoding = retriever.downsampler(query_encoding)
        query_encoding = F.normalize(query_encoding, dim=-1)
        scores = maxsim(query_encoding, doc_encodings.unsqueeze(0)).squeeze().tolist()
        ranking = np.argsort(scores)
        retrieved_chunks = [chunks[i] for i in ranking[:-11:-1]]

    # Re-identify
    length_correction = 0
    for i, mask_index in enumerate(local_mask_indices):
        local_text_read = local_text_read[:mask_index+length_correction] + "anon" + local_text_read[mask_index+6+length_correction:]
        length_correction -= 2

    ret_text = "\n\n".join(retrieved_chunks)

    prompt = prompt.format(retrieved=ret_text, text=local_text_read)

    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = reader_tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], reader, max_tokens=2048, temperature=0.1, eos_id=reader_tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = reader_tokenizer.decode(out_tokens[0])
    if "[MASK]" in result:
        print("="*30)
        print("Problem!")
        print(local_text_read)
        print(result)
    if len(local_text_read.split("[MASK]")) != 2:
        print("="*30)
        print("Problem!")
        print(local_text_read)
        print(local_mask_indices)
        print(result)
        print(mask_indices)
        print(mask_index_re_id)
        print(start_pos)
        print(end_pos)
        print(pos)

    result = result.replace("[MASK]", "")

    text = text[:mask_index_re_id] + result + text[mask_index_re_id+6:]

    correct = int(result == target)

    re_id_spans.append(result)

    target_tokens = reader_tokenizer.instruct_tokenizer.tokenizer.encode(target, False, False)

    correct_tokens, total_tokens = token_recall(target_tokens, out_tokens[0])

    order_id.append(pos)

    return text, correct, span_list, entity_type_list, entity_type, identifier_type_list, identifier_type, re_id_spans, correct_tokens, total_tokens, order_id


device = "cuda" if torch.cuda.is_available() else "cpu"

raws = [json.loads(line) for line in open("../datasets/Anonymized_TAB_with_general_plus_gens_rets.jsonl", "r")]

with open("../BMx_TAB/corpus.json", "r") as f:
    corpus = json.load(f)

nlp = spacy.load("en_core_web_trf")

query_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
doc_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

query_tokenizer.add_tokens("[Q]")
query_tokenizer.add_tokens("[ANON]")
doc_tokenizer.add_tokens("[D]")

retriever = ColBERT('google-bert/bert-base-cased', 768, 32, query_tokenizer, doc_tokenizer)
retriever.load_state_dict(torch.load("../models/colbert_wiki_all_mask.bin", map_location="cpu"), strict="True")
retriever.eval()
retriever.to(device)

mistral_models_path = Path("").joinpath('mistral_models', 'Nemo-Instruct')

reader_tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")
reader = Transformer.from_folder(mistral_models_path)

total_num_masks = 0
total_correct = 0
total_pred_tokens = 0
correct_pred_tokens = 0

correct_per_entity_type = defaultdict(int)
total_per_entity_type = defaultdict(int)
total_pred_tokens_per_entity_type = defaultdict(int)
correct_pred_tokens_per_entity_type = defaultdict(int)

correct_per_identifier_type = defaultdict(int)
total_per_identifier_type = defaultdict(int)
total_pred_tokens_per_identifier_type = defaultdict(int)
correct_pred_tokens_per_identifier_type = defaultdict(int)

re_identified_spans_per_doc = []
files_re_identified = []

progress_bar = tqdm(total=len(raws))

for i, datapoint in enumerate(raws):
    order_id = []

    file_to_re_id = datapoint["anonymized_text"]
    span_list = datapoint["masked_spans"][:]
    entity_type_list = datapoint["entity_type_list"][:]
    identifier_type_list = datapoint["identifier_type_list"][:]

    re_id_list = []

    mask_indices = []
    index_mask = file_to_re_id.find("[MASK]")
    while index_mask != -1:
        mask_indices.append(index_mask)
        index_mask = file_to_re_id.find("[MASK]", index_mask+6)

    total_num_masks += len(mask_indices)

    chunks, doc_encodings = generate_chunks_and_encoding(i, raws, doc_tokenizer, retriever, corpus, device, batch_size=32)

    num_correct = 0

    for _ in range(len(mask_indices)):
        file_to_re_id, correct, span_list, entity_type_list, entity_type, identifier_type_list, identifier_type, re_id_list, correct_tokens, total_tokens, order_id = re_identify_one_span(file_to_re_id, retriever, query_tokenizer, doc_encodings, chunks, reader, reader_tokenizer, 1, span_list, entity_type_list, identifier_type_list, re_id_list, order_id, device)
        num_correct += correct
        total_per_entity_type[entity_type] += 1
        correct_per_entity_type[entity_type] += correct
        total_per_identifier_type[identifier_type] += 1
        correct_per_identifier_type[identifier_type] += correct

        correct_pred_tokens += correct_tokens
        total_pred_tokens += total_tokens

        correct_pred_tokens_per_entity_type[entity_type] += correct_tokens
        total_pred_tokens_per_entity_type[entity_type] += total_tokens
        correct_pred_tokens_per_identifier_type[identifier_type] += correct_tokens
        total_pred_tokens_per_identifier_type[identifier_type] += total_tokens

    progress_bar.update()

    total_correct += num_correct

    files_re_identified.append(file_to_re_id)

    re_identified_spans = []
    j = -1
    for pos in order_id[::-1]:
        re_identified_spans.insert(pos, re_id_list[j])
        j -= 1

    re_identified_spans_per_doc.append(re_identified_spans)

    raws[i]["re-identified_file"] = file_to_re_id
    raws[i]["re-identifed_spans"] = re_identified_spans

progress_bar.close()

print(f"Exact Match: {(total_correct / total_num_masks) * 100:.2f}")
print(f"Token Recall: {(correct_pred_tokens / total_pred_tokens) * 100:.2f}")

for k, i in total_per_entity_type.items():
    print(f"Exact Match ({k}): {(correct_per_entity_type[k] / float(i)) * 100:.2f}")

for k, i in total_pred_tokens_per_entity_type.items():
    print(f"Token Recall ({k}): {(correct_pred_tokens_per_entity_type[k] / float(i)) * 100:.2f}")

for k, i in total_per_identifier_type.items():
    print(f"Exact Match ({k}): {(correct_per_identifier_type[k] / float(i)) * 100:.2f}")

for k, i in total_pred_tokens_per_identifier_type.items():
    print(f"Token Recall ({k}): {(correct_pred_tokens_per_identifier_type[k] / float(i)) * 100:.2f}")

with open("results_tab_general_zero_shot_run_3.txt", "w") as f:
    f.write(f"Exact Match: {(total_correct / total_num_masks) * 100:.2f}\n")
    f.write(f"Token Recall: {(correct_pred_tokens / total_pred_tokens) * 100:.2f}\n")
    f.write("\n")

    for k, i in total_per_entity_type.items():
        f.write(f"Exact Match ({k}): {(correct_per_entity_type[k] / float(i)) * 100:.2f}\n")

    for k, i in total_pred_tokens_per_entity_type.items():
        f.write(f"Token Recall ({k}): {(correct_pred_tokens_per_entity_type[k] / float(i)) * 100:.2f}\n")

    for k, i in total_per_identifier_type.items():
        f.write(f"Exact Match ({k}): {(correct_per_identifier_type[k] / float(i)) * 100:.2f}\n")

    for k, i in total_pred_tokens_per_identifier_type.items():
        f.write(f"Token Recall ({k}): {(correct_pred_tokens_per_identifier_type[k] / float(i)) * 100:.2f}\n")

with open("../datasets/Re-identified_TAB_General_Zero_Shot.jsonl", "w") as fo:
    for datapoint in raws:
        fo.write(json.dumps(datapoint))
        fo.write("\n")
