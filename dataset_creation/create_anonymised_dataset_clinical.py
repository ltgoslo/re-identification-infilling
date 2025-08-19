import json
import spacy
from tqdm.auto import tqdm
import re

nlp = spacy.load("en_core_web_trf")


if __name__ == "__main__":
    datapoint = {}
    dataset = []
    for line in tqdm(open("../datasets/Clinical_Dataset.jsonl", "r")):
        datapoint = {}
        note = json.loads(line)

        title = note["doc_id"]
        text = note["text"]

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

        dataset.append(datapoint)

    with open("../datasets/Anonymized_Clinical_notes.jsonl", "w") as fo:
        for d in dataset:
            print(json.dumps(d), file=fo)
