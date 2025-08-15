import json
import numpy as np
from baguetter.indices import BMXSparseIndex, SparseIndexConfig


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
    index = load_index("../BMx_General_TAB/bmx_index_background_tab", mmap=True)

    with open("../datasets/echr_test.json", "r") as f:
        raw = json.load(f)

    dataset = []
    datapoint = {}
    for case in raw:
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
        doc_id = f"{case['doc_id']}_case"

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

        q = datapoint["text"]
        ret = index.search(q).keys
        datapoint["retrieved_unmasked"] = ret
        q = datapoint["anonymized_text"]
        q = "".join(q.split("[MASK]"))
        ret = index.search(q).keys
        datapoint["retrieved_masked"] = ret

        dataset.append(datapoint)
        datapoint = {}

    with open("Anonymized_TAB_with_general_rets.jsonl", "w") as fo:
        for d in dataset:
            print(json.dumps(d), file=fo)
