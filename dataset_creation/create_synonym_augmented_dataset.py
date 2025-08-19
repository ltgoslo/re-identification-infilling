import json
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_retrieved_dataset', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Found at the following GitHub: https://github.com/Social-Data-inSights/coproximity_create_vocabulary
    with open("../../coproximity_create_vocabulary/vocabularyenglish/ngram_title_wiki/wiki_title_best_200000/synonyms.json", "r") as f:
        data = json.load(f)

    main_to_syn = defaultdict(list)
    for k, v in data.items():
        if v not in k:
            main_to_syn[v].append(k)

    documents = [json.loads(line) for line in open(args.sparse_retrieved_dataset, "r")]

    for document in documents:
        expanded_entities = []
        for entity in document["masked_spans"]:
            if entity in data.keys():
                main_token = data[entity]
                entities = main_to_syn[main_token][:]
                entities.append(main_token)
            elif entity in main_to_syn.keys():
                entities = main_to_syn[entity][:]
                entities.append(entity)
            else:
                entities = [entity]
            expanded_entities.append(entities)
        document["expanded_masked_spans"] = expanded_entities

    with open(args.output_file, "w") as fo:
        for d in documents:
            print(json.dumps(d), file=fo)
