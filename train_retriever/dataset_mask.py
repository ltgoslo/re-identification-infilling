import torch
from torch.utils.data import Dataset
from smart_open import open
import json
import random


class Dataset(Dataset):

    def __init__(self, file):

        self.data = []

        for line in open(file, "r"):
            self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        segment = self.data[index]["text"]
        for i, mask_index in enumerate(self.data[index]["mask_indices"]):
            segment = segment[:mask_index] + "[ANON]" + segment[mask_index+6:]

        segment = "[Q]" + segment
        positive_doc = "[D]" + self.data[index]["positive_retrievals"][random.randint(0, len(self.data[index]["positive_retrievals"])-1)]
        negative_doc = "[D]" + self.data[index]["negative_retrievals"][random.randint(0, len(self.data[index]["negative_retrievals"])-1)]

        labels = torch.zeros(1, dtype=torch.long)

        return segment, positive_doc, negative_doc, labels
