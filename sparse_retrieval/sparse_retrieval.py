import json
import numpy as np
from baguetter.indices import BMXSparseIndex, SparseIndexConfig
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_path', type=str, default="../sparse_indexes/BMx_TAB/bmx_index_background_tab")
    parser.add_argument('--documents', type=str, default="../datasets/echr_test.json")
    parser.add_argument("--output_file", type=str, default="../sparse_retrieved_datasets/Anonymized_TAB_with_general_rets.jsonl")

    args = parser.parse_args()
    return args


def load_index(
    path: str,
    *,
    mmap: bool = False,
) -> BMXSparseIndex:
    """Load an index from the given path or name.

    Args:
        path (str): Name or path of the index.
        repository (AbstractFileRepository): File repository to load from.
        mmap (bool): Whether to memory-map the file.

    Returns:
        BaseSparseIndex: The loaded index.

    Raises:
        FileNotFoundError: If the index file is not found.

    """

    mmap_mode = "r" if mmap else None
    with open(path, "rb") as f:
        stored = np.load(f, allow_pickle=True, mmap_mode=mmap_mode)
        state = stored["state"][()]
        retriever = BMXSparseIndex()
        retriever = retriever.from_config(SparseIndexConfig(**state["config"]))
        retriever.key_mapping = state["key_mapping"]
        retriever.index = state["index"]
        retriever.corpus_tokens = state["corpus_tokens"]
        return retriever


if __name__ == "__main__":
    args = parse_args()

    index = load_index(args.index_path, mmap=True)

    with open(args.documents, "r") as f:
        docs = json.load(f)

    dataset = []
    datapoint = {}
    for doc in docs:
        span_list = []
        entity_type_list = []
        identifier_type_list = []
        starts = []
        ends = []

        if "annotations" in doc:  # If the document has manual annotations in the style of TAB.
            annotator = np.random.choice(list(doc["annotations"].keys()), 1)[0]
            for entity in doc["annotations"][annotator]["entity_mentions"]:
                if entity["identifier_type"] != "NO_MASK":
                    span_list.append(entity["span_text"])
                    entity_type_list.append(entity["entity_type"])
                    identifier_type_list.append(entity["identifier_type"])
                    starts.append(entity["start_offset"])
                    ends.append(entity["end_offset"])
        else:
            pass

        text = doc["text"]
        doc_id = f"{doc['doc_id']}_case"

        anonymized_text = ""
        cur = 0
        for i, start in enumerate(starts):
            anonymized_text += text[cur:start] + "[MASK]"
            cur = ends[i]
        anonymized_text += text[cur:]

        datapoint["doc_id"] = doc_id
        datapoint["text"] = text
        datapoint["anonymized_text"] = anonymized_text
        if "annotations" in doc:  # If the document has manual annotations in the style of TAB.
            datapoint["masked_spans"] = span_list
            datapoint["entity_type_list"] = entity_type_list
            datapoint["identifier_type_list"] = identifier_type_list
        else:
            datapoint["masked_spans"] = doc["masked_spans"]

        q = datapoint["text"]
        ret = index.search(q).keys
        datapoint["retrieved_unmasked"] = ret
        q = datapoint["anonymized_text"]
        q = "".join(q.split("[MASK]"))
        ret = index.search(q).keys
        datapoint["retrieved_masked"] = ret

        dataset.append(datapoint)
        datapoint = {}

    with open(args.output_file, "w") as fo:
        for d in dataset:
            print(json.dumps(d), file=fo)
