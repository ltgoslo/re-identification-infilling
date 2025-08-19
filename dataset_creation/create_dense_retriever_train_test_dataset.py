import json
from tqdm.auto import tqdm
import re
from collections import defaultdict
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_retrieved_dataset', type=str, default="../datasets/Anonymized_bios_train_with_rets_expanded.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--background_corpus", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    documents = [json.loads(line) for line in open(args.sparse_retrieved_dataset, "r")]
    corpus = json.load(open(args.background_corpus, "r"))

    train_data = []
    chunks = []
    chunk_len = 600
    for document in tqdm(documents):
        for ret_id in document["retrieved_documents"]:
            start = 0
            ret_text = corpus[ret_id]
            while len(ret_text) - start > chunk_len:
                chunk = ret_text[start: start + chunk_len]
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
            chunks.append(ret_text[start:])

        positive_passages = defaultdict(list)
        negative_passages = defaultdict(list)

        for synonyms, entity in zip(document["expanded_masked_spans"], document["masked_spans"]):  # The synonyms list contains the original entity
            if entity in positive_passages:
                continue
            synonym_match = False
            for synonym in synonyms:
                if synonym in positive_passages:
                    positive_passages[entity] = positive_passages[synonym]
                    negative_passages[entity] = negative_passages[synonym]
                    synonym_match = True
                    break
            if synonym_match:
                continue
            for chunk in chunks:
                added = False
                for synonym in synonyms:
                    if re.search(r"\W{}\W".format(synonym), chunk) is not None:
                        positive_passages[entity].append(chunk)
                        added = True
                        break
                    elif re.search(r"^{}\W".format(synonym), chunk) is not None:
                        positive_passages[entity].append(chunk)
                        added = True
                        break
                    elif re.search(r"\W{}$".format(synonym), chunk) is not None:
                        positive_passages[entity].append(chunk)
                        added = True
                        break
                    elif re.search(r"^{}$".format(synonym), chunk) is not None:
                        positive_passages[entity].append(chunk)
                        added = True
                        break
                if not added:
                    negative_passages[entity].append(chunk)

        text = document["anonymized_text"]

        mask_indices = []
        index_mask = text.find("[MASK]")
        while index_mask != -1:
            mask_indices.append(index_mask)
            index_mask = text.find("[MASK]", index_mask+6)

        masked_ents = document["masked_spans"]

        for i, (mask_index, entity) in enumerate(zip(mask_indices, masked_ents)):
            if entity not in positive_passages:
                continue

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

            local_mask_ents = []
            local_mask_indices = []
            for j, local_mask_index in enumerate(mask_indices):
                if local_mask_index > end_pos:
                    break
                if local_mask_index >= start_pos and j != i:
                    local_mask_ents.append(masked_ents[j])
                    local_mask_indices.append(local_mask_index-start_pos)

            random.shuffle(positive_passages[entity])
            random.shuffle(negative_passages[entity])

            train_data.append(
                {
                    "text": local_text,
                    "target": entity,
                    "target_index": mask_index-start_pos,
                    "mask_ents": local_mask_ents,
                    "mask_indices": local_mask_indices,
                    "positive_retrievals": positive_passages[entity][:15],
                    "negative_retrievals": negative_passages[entity][:15]
                }
            )

    with open(args.output_file, "r") as output_file:
        for datapoint in train_data:
            print(json.dumps(datapoint), file=output_file)
