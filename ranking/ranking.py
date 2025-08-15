from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from metrics import top_k_accuracy, calculate_mrr
from dataset_ranker import InputDataset, NegativeReducedDataset
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--texts_to_rank", required=True, type=str, help="Name of the pickled list containing all texts to rank, excepted to be a pickled list in the same order as the candidates.")
    parser.add_argument("--candidate_list", required=True, type=str, help="Name of the pickled list containing all the names of the candidates.")
    parser.add_argument("--ranker_model", required=True, type=str, help="The path to ranker model weights. We assume the model is trained based on bert-base-cased from huggingface.")
    parser.add_argument("--seed", default=782364, type=int, help="Random number generator seed.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size to pass to the ranker.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.texts_to_rank, "rb") as f:
        texts = pickle.load(f)

    with open(args.candidate_list, "rb") as f:
        candidates = pickle.load(f)

    unique_labels = list(set(candidates))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)

    DEV_BATCH_SIZE = args.batch_size

    dev_dataset_pos = InputDataset(texts, candidates, percentage_true_labels=1.0)
    dev_dataset_neg = NegativeReducedDataset(texts, candidates, num_negative_labels=len(unique_labels)-1, unique_labels=unique_labels)

    dev_dataloader_pos = DataLoader(dev_dataset_pos, batch_size=DEV_BATCH_SIZE, shuffle=False)
    dev_dataloader_neg = DataLoader(dev_dataset_neg, batch_size=DEV_BATCH_SIZE, shuffle=False)

    print("########### Starting Ranking ############")

    ranker_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    ranker_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
    ranker_model.load_state_dict(torch.load(args.ranker_model, map_location="cpu", weights_only=True))
    ranker_model.to(device)
    ranker_model.eval()

    ranks = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader_neg):
            tokens = ranker_tokenizer(batch[0], text_pair=batch[1], truncation=True, padding=True, max_length=512, return_tensors="pt")
            logits = ranker_model(**tokens.to(device)).logits
            for j, text in enumerate(batch[0]):
                ranks[text].append((logits[j].item(), "negative", batch[1][j]))

        for batch, labels in dev_dataloader_pos:
            tokens = ranker_tokenizer(batch[0], text_pair=batch[1], truncation=True, padding=True, max_length=512, return_tensors="pt")
            logits = ranker_model(**tokens.to(device)).logits
            for j, text in enumerate(batch[0]):
                ranks[text].append((logits[j].item(), "positive", batch[1][j]))

    top_1_acc = top_k_accuracy(ranks, k=1)
    top_5_acc = top_k_accuracy(ranks, k=5)
    top_10_acc = top_k_accuracy(ranks, k=10)
    mrr = calculate_mrr(ranks)

    print(f"Development top-1 accuracy: {top_1_acc}; top-5 accuracy: {top_5_acc}; top-10 accuracy: {top_10_acc}; mrr: {mrr}")
