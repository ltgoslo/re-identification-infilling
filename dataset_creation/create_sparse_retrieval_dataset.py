from baguetter.indices import BMXSparseIndex, SparseIndexConfig
import json
import numpy as np
from tqdm.auto import tqdm
import argparse
import re


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--anonymized_documents', type=str, required=True)
    parser.add_argument("--ouptut_file", type=str, required=True)
    parser.add_argument("--retrieved_on_unmasked", action="store_true", default=False)
    parser.add_argument("--include_original", action="store_true", help="Whether to keep the original text if retrieved.")
    parser.add_argument("--num_doc_retrieved", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    index = load_index(args.index, mmap=True)

    documents = [json.loads(line) for line in open(args.anonymized_documents, "r")]

    for doc in tqdm(documents):
        if args.retrieved_on_unmasked:
            query = doc["text"]
        else:
            query = doc["anonymized_text"]
            query = re.sub(r"\s*\[MASK\]", "", query)
        ret = index.search(query, top_k=args.num_doc_retrieved+1).keys
        if args.include_original:
            ret = ret[:args.num_doc_retrieved]
        elif doc["doc_id"] in ret:
            ret.remove(doc["doc_id"])
        else:
            ret = ret[:args.num_doc_retrieved]
        doc["retrieved_documents"] = ret

    with open(args.output_file, "w") as fo:
        for doc in documents:
            print(json.dumps(doc), file=fo)
