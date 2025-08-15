from baguetter.indices import BMXSparseIndex
import json
import numpy as np
import dataclasses
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--background_dataset', type=str, required=True)
    parser.add_argument('--sparse_index_dir', type=str, required=True)
    parser.add_argument("--sparse_index_name", type=str, required=True)

    args = parser.parse_args()
    return args


def save_index(
    index,
    path: str,
) -> str:
    """Save the index to the given path.

    Args:
        path (str): Path to save the index to.
        repository (AbstractFileRepository): File repository to save to.

    """
    state = {
        "key_mapping": index.key_mapping,
        "index": index.index,
        "corpus_tokens": index.corpus_tokens,
        "config": dataclasses.asdict(index.config),
    }
    with open(path, "wb") as f:
        np.savez_compressed(f, state=state)
    return path


if __name__ == "__main__":
    args = parse_args()

    dataset = []
    for line in open(args.background_dataset, "r"):
        dataset.append(json.loads(line))

    docs = [d["doc"] for d in dataset]
    doc_ids = [d["id"] for d in dataset]

    dataset = {k: v for k, v in zip(doc_ids, docs)}

    with open(f"{args.sparse_index_dir}/corpus.json", "w") as f:
        json.dump(dataset, f)

    index = BMXSparseIndex()
    index.add_many(doc_ids, docs, show_progress=True)

    save_index(index, f"{args.sparse_index_dir}/{args.sparse_index_name}")
