import json
from tqdm.auto import tqdm
import numpy as np


if __name__ == "__main__":
    with open("../datasets/echr_test.json", "r") as f:
        cases = json.load(f)

    dataset = []
    datapoint = {}
    for case in tqdm(cases):
        span_list = []
        entity_type_list = []
        identifier_type_list = []
        starts = []
        ends = []

        annotator = np.random.choice(list(case["annotations"].keys()), 1)[0]
        for entity in case["annotations"][annotator]["entity_mentions"]:
            if entity["identifier_type"] != "NO_MASK":
                span_list.append(entity["span_text"])
                entity_type_list.append(entity["entity_type"])
                identifier_type_list.append(entity["identifier_type"])
                starts.append(entity["start_offset"])
                ends.append(entity["end_offset"])

        text = case["text"]
        doc_id = case["doc_id"]

        anonymized_text = ""
        cur = 0
        for i, start in enumerate(starts):
            anonymized_text += text[cur:start] + "[MASK]"
            cur = ends[i]
        anonymized_text += text[cur:]

        datapoint["doc_id"] = doc_id
        datapoint["text"] = text
        datapoint["anonymized_text"] = anonymized_text
        datapoint["masked_spans"] = span_list
        datapoint["entity_type_list"] = entity_type_list
        datapoint["identifier_type_list"] = identifier_type_list

    with open("../datasets/Anonymized_TAB_cases.jsonl", "w") as fo:
        for d in dataset:
            print(json.dumps(d), file=fo)
