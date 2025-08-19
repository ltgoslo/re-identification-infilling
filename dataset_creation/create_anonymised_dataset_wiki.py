import json
import spacy
import random
from tqdm.auto import tqdm
import re

nlp = spacy.load("en_core_web_trf")


if __name__ == "__main__":
    datapoint = {}
    dataset_train = []
    dataset_dev = []
    dataset_test = []
    for line in tqdm(open("../datasets/bios.jsonl", "r")):
        datapoint = {}
        if random.random() < 0.9985:
            continue
        bio = json.loads(line)

        title = bio["id"]
        text = bio["contents"]

        anonymized_text = ""
        entities = []

        text = re.sub(r"\('\)", "", text)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r" +", " ", text)

        doc = nlp(text)

        cur = 0
        for ent in doc.ents:
            span = ent.text
            start = ent.start_char
            end = ent.end_char
            anonymized_text += text[cur:start]
            entities.append(span)
            anonymized_text += "[MASK]"
            cur = end
        anonymized_text += text[cur:]

        datapoint["title"] = title
        datapoint["text"] = text
        datapoint["anonymized_text"] = anonymized_text
        datapoint["masked_spans"] = entities

        split = random.randint(1, 10)
        if split < 9:
            dataset_train.append(datapoint)
        elif split == 9:
            dataset_dev.append(datapoint)
        else:
            dataset_test.append(datapoint)

    with open("../datasets/Anonymized_bios_train.jsonl", "w") as fo:
        for d in dataset_train:
            print(json.dumps(d), file=fo)
    with open("../datasets/Anonymized_bios_dev.jsonl", "w") as fo:
        for d in dataset_dev:
            print(json.dumps(d), file=fo)
    with open("../datasets/Anonymized_bios_test.jsonl", "w") as fo:
        for d in dataset_test:
            print(json.dumps(d), file=fo)
