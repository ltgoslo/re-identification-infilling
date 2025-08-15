from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from metrics import top_k_accuracy, calculate_mrr
from dataset_ranker import InputDataset, NegativeReducedDataset
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

with open("../datasets/clinical_all_mistral.pkl", "rb") as f:
    re_id_articles = pickle.load(f)

with open("../datasets/clinical_names2.pkl", "rb") as f:
    dev_labels = pickle.load(f)

with open("../datasets/clinical_masked.pkl", "rb") as f:
    masked_articles = pickle.load(f)

unique_labels = list(set(dev_labels))

print(len(unique_labels))

device = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(782364)

DEV_BATCH_SIZE = 512

dev_dataset_pos = InputDataset(masked_articles, dev_labels, percentage_true_labels=1.0)
dev_dataset_neg = NegativeReducedDataset(masked_articles, dev_labels, num_negative_labels=len(unique_labels)-1, unique_labels=unique_labels)

dev_dataloader_pos = DataLoader(dev_dataset_pos, batch_size=DEV_BATCH_SIZE, shuffle=False)
dev_dataloader_neg = DataLoader(dev_dataset_neg, batch_size=DEV_BATCH_SIZE, shuffle=False)

print("########### Before Re-identification ############")

ranker_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
ranker_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
ranker_model.load_state_dict(torch.load("../models/ranker_model.pt", map_location="cpu", weights_only=True))
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

print("########### After Re-identification ############")

dev_dataset_pos = InputDataset(re_id_articles, dev_labels, percentage_true_labels=1.0)
dev_dataset_neg = NegativeReducedDataset(re_id_articles, dev_labels, num_negative_labels=len(unique_labels)-1, unique_labels=unique_labels)

dev_dataloader_pos = DataLoader(dev_dataset_pos, batch_size=DEV_BATCH_SIZE, shuffle=False)
dev_dataloader_neg = DataLoader(dev_dataset_neg, batch_size=DEV_BATCH_SIZE, shuffle=False)

ranks = defaultdict(list)
scores_pos = []
scores_neg = []

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
