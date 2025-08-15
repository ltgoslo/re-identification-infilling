import torch
from torch.utils.data import Dataset
from smart_open import open
from transformers.tokenization_utils_base import BatchEncoding
import json
import random


# Adapted from https://huggingface.co/THUDM/glm-roberta-large/blob/main/tokenization_glm.py
def build_inputs_for_generation(tokenizer, model_input: BatchEncoding, max_gen_length=512, targets=None, padding=False):
    mask_ids = tokenizer.mask_token_ids
    input_ids = model_input.input_ids
    batch_size, seq_length = input_ids.shape[:2]
    position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
    position_ids, block_position_ids = [], []
    labels = None
    if targets is not None:
        is_batched = isinstance(targets, (list, tuple))
        targets = tokenizer(targets, add_special_tokens=False, padding=False).input_ids
        if not is_batched:
            targets = [targets]
        assert len(targets) == len(input_ids)
        targets = [(target + [tokenizer.eop_token_id])[:max_gen_length] for target in targets]
        if not padding:
            max_gen_length = max(map(len, targets))
        targets = [[tokenizer.sop_token_id] + target for target in targets]
        labels = [target[1:] for target in targets]
        targets = [target + [tokenizer.pad_token_id] * (max_gen_length + 1 - len(target)) for target in targets]
        labels = [label + [-100] * (max_gen_length - len(label)) for label in labels]
        targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
        labels = torch.tensor(labels, dtype=input_ids.dtype, device=input_ids.device)
        labels = torch.cat((input_ids.new_full((batch_size, seq_length), -100), labels), dim=1)
    for i in range(batch_size):
        mask_positions = []
        for mask_id in mask_ids:
            mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
        if not mask_positions:
            print(input_ids[i])
            print(mask_positions)
            print(tokenizer.decode(input_ids[i]))
            raise ValueError("Cannot find mask token in the input")
        mask_positions.sort()
        mask_pos = mask_positions[0]
        position_ids.append(position_id + [mask_pos] * max_gen_length)
        block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
    position_ids = torch.tensor(position_ids, dtype=input_ids.dtype, device=input_ids.device)
    block_position_ids = torch.tensor(block_position_ids, dtype=input_ids.dtype, device=input_ids.device)
    position_ids = torch.stack((position_ids, block_position_ids), dim=1)
    attention_mask = model_input.attention_mask
    attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
    generation_attention_mask = torch.cat(
        [
            attention_mask.new_zeros((seq_length, max_gen_length)),
            torch.tril(attention_mask.new_ones((max_gen_length, max_gen_length)))
        ],
        dim=0
    ).unsqueeze(0).expand(batch_size, -1, -1)
    attention_mask = torch.cat((attention_mask, generation_attention_mask), dim=2)
    attention_mask = attention_mask.unsqueeze(1)
    if targets is None:
        input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), tokenizer.sop_token_id)), dim=-1)
    else:
        input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
    batch = {"input_ids": input_ids, "position_ids": position_ids}
    if labels is None:
        batch["generation_attention_mask"] = attention_mask
    else:
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
    return BatchEncoding(batch)


class Dataset(Dataset):

    def __init__(self, file, unmask_probability=0.1):

        self.unmask_probability = unmask_probability

        self.data = []

        for line in open(file, "r"):
            self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        unmask_probability = random.uniform(0, 1)
        unmask_probability = 0.0
        target = self.data[index]["target"]
        segment = self.data[index]["text"]
        length_correction = 0
        for i, mask_index in enumerate(self.data[index]["mask_indices"]):
            if random.random() < unmask_probability:
                segment = segment[:mask_index+length_correction] + self.data[index]["mask_ents"][i] + segment[mask_index+6+length_correction:]
                length_correction = len(self.data[index]["mask_ents"][i]) - 6 + length_correction
            else:
                segment = segment[:mask_index+length_correction] + "anon" + segment[mask_index+6+length_correction:]
                length_correction -= 2

        retrieved_chunk = self.data[index]["retrievals"][0]

        return segment, retrieved_chunk, target
