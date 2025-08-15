from torch.utils.data import Dataset
import numpy as np


class InputDataset(Dataset):

    def __init__(self, data, labels, percentage_true_labels=0.5):
        self.data = data
        self.labels = labels
        self.percentage_true_labels = percentage_true_labels
        self.data_len = len(data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        true_label = np.random.rand() < self.percentage_true_labels
        if true_label:
            label_id = idx
        else:
            label_id = int(np.random.choice(self.data_len - 1, 1))
            if label_id >= idx and idx != self.data_len - 1:
                label_id += 1
            elif label_id >= idx and idx == self.data_len - 1:
                label_id = 0

        return (self.data[idx], self.labels[label_id]), int(true_label)


class NegativeReducedDataset(Dataset):

    def __init__(self, data, labels, num_negative_labels=50):
        self.data = []
        self.labels = []
        data_len = len(data)
        for i, d in enumerate(data):
            label_ids = np.random.choice(data_len, min(num_negative_labels, data_len), replace=False)
            for label_id in label_ids:
                if label_id >= i and i != (data_len - 1) and label_id != (data_len - 1):
                    self.data.append(d)
                    self.labels.append(labels[label_id+1])
                elif label_id >= i and (i == (data_len - 1) or label_id == (data_len - 1)):
                    self.data.append(d)
                    self.labels.append(labels[0])
                else:
                    self.data.append(d)
                    self.labels.append(labels[label_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TrainDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_id = int(np.random.choice(len(self.data), 1))
        while label_id == idx:
            label_id = int(np.random.choice(len(self.data), 1))
        return self.data[idx], self.labels[idx], self.labels[label_id]
