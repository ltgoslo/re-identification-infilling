import json
from tqdm.auto import tqdm
import re
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer
from colbert import ColBERT, maxsim
import torch.nn.functional as F


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sparse_retrieved_dataset", required=True, type=str)
    parser.add_argument("--path_to_colbert_model", required=True, type=str)
    parser.add_argument("--background_data_path", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)

    args = parser.parse_args()

    return args


def generate_chunks_and_encoding(doc_id, retrieved, doc_tokenizer, model, corpus, device, batch_size=16):
    chunks = []
    chunk_len = 600
    for ret in retrieved[doc_id]["retrieved_documents"]:
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


def get_retrieved_for_text(doc_id, text, masked_ents, ents, chunks, doc_encodings, query_tokenizer, model, dataset, device, batch_size=16):
    mask_indices = []
    index_mask = text.find("[MASK]")
    while index_mask != -1:
        mask_indices.append(index_mask)
        index_mask = text.find("[MASK]", index_mask+6)
    segments = []
    local_texts = []
    mi = []
    start_poses = []
    all_local_mask_ents = []
    all_local_mask_indices = []
    targets = []
    evaluated = False
    for i, mask_index in enumerate(mask_indices):
        evaluated = False
        local_text = text[max(0, mask_index-200):min(len(text), mask_index+206)]

        intial_space = local_text.find(" ") if mask_index-200 > 0 else -1
        final_space = 206
        while local_text.find(" ", final_space+1) != -1:
            final_space = local_text.find(" ", final_space+1)

        start_pos = intial_space+1 + max(0, mask_index-200)
        end_pos = final_space + max(0, mask_index-200) if len(text) > mask_index+206 else len(local_text) - 1
        if final_space > mask_index - start_pos and mask_index-start_pos >= 0:
            local_text = local_text[intial_space+1:final_space]
        else:
            print("Problem!")
            print(local_text)
            print(final_space)
            print(mask_index - start_pos)
            start_pos = max(0, mask_index-200)
            end_pos = max(0, mask_index-200) + len(local_text)

        mi.append(mask_index)
        start_poses.append(start_pos)

        local_mask_ents = []
        local_mask_indices = []
        for j, local_mask_index in enumerate(mask_indices):
            if local_mask_index > end_pos:
                break
            if local_mask_index >= start_pos and j != i:
                local_mask_ents.append(masked_ents[j])
                local_mask_indices.append(local_mask_index-start_pos)

        target = masked_ents[i]
        targets.append(target)
        all_local_mask_ents.append(local_mask_ents)
        all_local_mask_indices.append(local_mask_indices)

        for i, mask_index in enumerate(local_mask_indices):
            local_text = local_text[:mask_index+0] + "[ANON]" + local_text[mask_index+6+0:]

        local_texts.append(local_text)
        segments.append("[Q]" + local_text)

        if len(segments) != batch_size:
            continue

        with torch.no_grad():
            query = query_tokenizer(segments, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
            query.attention_mask[:, :] = 1
            query_encoding = model.query_encoder(query["input_ids"], query["attention_mask"]).last_hidden_state
            query_encoding = model.downsampler(query_encoding)
            query_encoding = F.normalize(query_encoding, dim=-1)
            scores = maxsim(query_encoding.cpu(), doc_encodings.unsqueeze(0).cpu()).squeeze().tolist()
        for j, score in enumerate(scores):
            ranking = np.argsort(score)
            retrieved_chunks = [chunks[i] for i in ranking[:-11:-1]]

            dataset.append({"text": local_texts[j], "target": targets[j], "target_index": mi[j]-start_poses[j], "mask_ents": all_local_mask_ents[j], "mask_indices": all_local_mask_indices[j], "retrievals": retrieved_chunks})

        segments = []
        local_texts = []
        mi = []
        start_poses = []
        all_local_mask_ents = []
        all_local_mask_indices = []
        targets = []
        evaluated = True

    if not evaluated and segments:
        with torch.no_grad():
            query = query_tokenizer(segments, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(device)
            query.attention_mask[:, :] = 1
            query_encoding = model.query_encoder(query["input_ids"], query["attention_mask"]).last_hidden_state
            query_encoding = model.downsampler(query_encoding)
            query_encoding = F.normalize(query_encoding, dim=-1)
            scores = maxsim(query_encoding.cpu(), doc_encodings.unsqueeze(0).cpu()).tolist()
        for i, score in enumerate(scores):
            ranking = np.argsort(score)
            retrieved_chunks = [chunks[i] for i in ranking[:-11:-1]]

            dataset.append({"text": local_texts[i], "target": targets[i], "target_index": mi[i]-start_poses[i], "mask_ents": all_local_mask_ents[i], "mask_indices": all_local_mask_indices[i], "retrievals": retrieved_chunks})

    return dataset


if __name__ == "__main__":
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrievals = [json.loads(line) for line in open(args.sparse_retrieved_dataset, "r")]

    query_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    doc_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

    query_tokenizer.add_tokens("[Q]")
    query_tokenizer.add_tokens("[ANON]")
    doc_tokenizer.add_tokens("[D]")

    model = ColBERT('google-bert/bert-base-cased', 768, 32, query_tokenizer, doc_tokenizer)

    model.load_state_dict(torch.load(args.path_to_colbert_model, map_location="cpu"), strict="True")
    model.eval()

    model.to(device)

    dataset = []

    args = parse_arguments()

    with open(args.background_data_path, "r") as f:
        corpus = json.load(f)

    start_index = args.start_index

    if start_index > len(retrievals):
        print("Error start is past len of retrievals!")
        quit()

    if args.end_index == -1 or args.end_index > len(retrievals):
        end_index = len(retrievals)
    else:
        end_index = args.end_index

    print(start_index, end_index)

    progress_bar = tqdm(total=end_index - start_index)

    for doc_id in range(start_index, end_index):
        if not retrievals[doc_id]["retrieved_documents"]:
            continue

        chunks, doc_encodings = generate_chunks_and_encoding(doc_id, retrievals, doc_tokenizer, model, corpus, device, batch_size=64)

        text = retrievals[doc_id]["anonymized_text"]
        masked_ents = retrievals[doc_id]["masked_spans"]
        ents = retrievals[doc_id]["expanded_masked_spans"]

        dataset = get_retrieved_for_text(doc_id, text, masked_ents, ents, chunks, doc_encodings, query_tokenizer, model, dataset, device, batch_size=32)

        progress_bar.update()

    progress_bar.close()

    with open(args.output_file, "w") as fo:
        for datapoint in dataset:
            fo.write(json.dumps(datapoint))
            fo.write("\n")
