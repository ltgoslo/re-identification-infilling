import numpy as np


def top_k_accuracy(scores, k=5):
    top_k_acc = np.zeros(len(scores))
    for j, (_, score) in enumerate(scores.items()):
        for i, (_, label, _) in enumerate(sorted(score, reverse=True)[:k]):
            if label == "positive":
                top_k_acc[j] = 1
                break

    return np.mean(top_k_acc)


def calculate_mrr(scores):
    mrr = np.zeros(len(scores))
    for j, (_, score) in enumerate(scores.items()):
        for i, (_, label, _) in enumerate(sorted(score, reverse=True)[:]):
            if label == "positive":
                mrr[j] = 1/(i+1)
                break

    return np.mean(mrr)
