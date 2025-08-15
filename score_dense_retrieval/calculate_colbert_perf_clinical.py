# Torch
import torch
import torch.nn.functional as F

# I/O
import json
import argparse
from smart_open import open

# Text manipulation
import re

# Tansformers / Models
from transformers import AutoTokenizer
from colbert import ColBERT, maxsim

# Utils
import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="../datasets/Anonymized_Clinical_notes_with_all_rets.jsonl")
    parser.add_argument('--output_file', type=str, default="../datasets/ColBERT_Performance_clinical_notes_all.jsonl")
    parser.add_argument("--colbert_model", type=str, default="../models/colbert_wiki_all_mask.bin")

    args = parser.parse_args()
    return args


def generate_chunks_and_encoding(doc_id, retrieved, doc_tokenizer, model, corpus, device, batch_size=16):
    chunks = []
    chunk_len = 600
    for ret in retrieved[doc_id]["retrieved_masked"]:
        doc = corpus[ret]
        start = 0
        while len(doc) - start > chunk_len:
            chunk = doc[start:start+chunk_len]
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
        chunks.append(doc[start:])

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


def retrieve(text, retriever, query_tokenizer, doc_encodings, chunks, mask_index, target, mas_indices, curr_pos, device="cpu", num_char=200):
    local_text = text[max(0, mask_index-num_char):min(len(text), mask_index+num_char+6)]

    intial_space = local_text.find(" ") if mask_index-num_char > 0 else -1
    final_space = num_char + 6
    while local_text.find(" ", final_space+1) != -1:
        final_space = local_text.find(" ", final_space+1)

    start_pos = intial_space+1 + max(0, mask_index-num_char)
    end_pos = final_space + max(0, mask_index-num_char) if len(text) > mask_index+num_char+6 else len(local_text) - 1
    if final_space > mask_index - start_pos and mask_index-start_pos >= 0:
        local_text = local_text[intial_space+1:final_space]
    else:
        print("Problem!")
        print(local_text)
        print(final_space)
        print(mask_index - start_pos)
        start_pos = max(0, mask_index-num_char)
        end_pos = max(0, mask_index-num_char) + len(local_text)

    local_mask_indices = []
    for j, local_mask_index in enumerate(mask_indices):
        if local_mask_index > end_pos:
            break
        if local_mask_index >= start_pos and j != curr_pos:
            local_mask_indices.append(local_mask_index-start_pos)

    for i, mask_index in enumerate(local_mask_indices):
        local_text = local_text[:mask_index+0] + "[ANON]" + local_text[mask_index+6+0:]

    with torch.no_grad():
        segment = "[Q]" + local_text
        query = query_tokenizer(segment, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
        query.attention_mask[:, :] = 1
        query_encoding = retriever.query_encoder(query["input_ids"], query["attention_mask"]).last_hidden_state
        query_encoding = retriever.downsampler(query_encoding)
        query_encoding = F.normalize(query_encoding, dim=-1)
        scores = maxsim(query_encoding, doc_encodings.unsqueeze(0)).squeeze().tolist()
        ranking = np.argsort(scores)
        retrieved_chunks = [chunks[i] for i in ranking[::-1]]

    pos = None
    for i, chunk in enumerate(retrieved_chunks):
        if target in chunk:
            pos = i + 1
            break

    return local_text, pos, retrieved_chunks[:5]


args = parse_args()

with open("../BMx_Clinical/corpus.json", "r") as f:
    corpus = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

raws = [json.loads(line) for line in open(args.input_file, "r")]

query_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
doc_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

query_tokenizer.add_tokens("[Q]")
query_tokenizer.add_tokens("[ANON]")
doc_tokenizer.add_tokens("[D]")

retriever = ColBERT('google-bert/bert-base-cased', 768, 32, query_tokenizer, doc_tokenizer)
retriever.load_state_dict(torch.load(args.colbert_model, map_location="cpu", weights_only=True), strict="True")
retriever.eval()
retriever.to(device)

colbert_perf = []

progress_bar = tqdm(total=len(raws))

for i, datapoint in enumerate(raws):
    curr = {}

    text = datapoint["anonymized_text"]
    span_list = datapoint["masked_spans"][:]

    mask_indices = []
    index_mask = text.find("[MASK]")
    while index_mask != -1:
        mask_indices.append(index_mask)
        index_mask = text.find("[MASK]", index_mask+6)

    chunks, doc_encodings = generate_chunks_and_encoding(i, raws, doc_tokenizer, retriever, corpus, device, batch_size=32)

    for i, mask_index in enumerate(mask_indices):
        target = span_list[i]
        local_text, pos, retrieved_chunks = retrieve(text, retriever, query_tokenizer, doc_encodings, chunks, mask_index, target, mask_indices, i, device=device)

        curr["doc_id"] = datapoint["doc_id"]
        curr["local_text"] = local_text
        curr["pos"] = pos
        curr["retrieved_chunks"] = retrieved_chunks

        colbert_perf.append(curr)
    progress_bar.update()

progress_bar.close()

with open(args.output_file, "w") as fo:
    for datapoint in colbert_perf:
        fo.write(json.dumps(datapoint))
        fo.write("\n")

total_pos = 0
total_exist = 0
poses = []
total_at_1 = 0
total_at_5 = 0
total_at_10 = 0
mrrs = []
for p in colbert_perf:
    if p["pos"]:
        total_pos += p["pos"]
        poses.append(p["pos"])
        mrrs.append(1/float(p["pos"]))
        total_exist += 1
        if p["pos"] == 1:
            total_at_1 += 1
        if p["pos"] < 6:
            total_at_5 += 1
        if p["pos"] < 11:
            total_at_10 += 1

print(f"MRR: {np.mean(mrrs)}; Acc@1: {total_at_1 / total_exist}; Acc@5: {total_at_5 / total_exist}; Acc@10: {total_at_5 / total_exist}")
