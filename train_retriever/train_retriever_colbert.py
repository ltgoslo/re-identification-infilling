import torch
from transformers import AutoTokenizer
from dataset import Dataset
from colbert import ColBERT
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import argparse
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a PLM on the LAMA knowledge probe')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size to use during probing')
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help='The batch size to use during probing')
    parser.add_argument('--model_name', type=str,
                        default='google-bert/bert-base-cased', help='The pretrained model to use')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='The subset of LAMA to probe on')
    parser.add_argument('--compress_size', type=int, default=32,
                        help='The subset of LAMA to probe on')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='The subset of LAMA to probe on')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='The subset of LAMA to probe on')
    parser.add_argument('--num_warmup_steps', type=int, default=1000,
                        help='The subset of LAMA to probe on')
    parser.add_argument('--seed', type=int, default=42, help='The rng seed')

    args = parser.parse_args()
    return args


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(data):
    texts = [datapoint[0] for datapoint in data]
    positive_documents = [datapoint[1] for datapoint in data]
    negative_documents = [datapoint[2] for datapoint in data]
    labels = torch.cat([datapoint[3] for datapoint in data])
    return (texts, positive_documents, negative_documents, labels)


args = parse_args()

NUM_STEPS = args.num_steps
BATCH_SIZE = args.batch_size
LR = args.learning_rate

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    wandb.init(
        name="ColBERT",
        project="PROJECT",
        entity="NAME",
        allow_val_change=True,
        reinit=True,
    )

dataset = Dataset("../datasets/Retrieval_train_dataset.jsonl")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

query_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
doc_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

query_tokenizer.add_tokens("[Q]")
query_tokenizer.add_tokens("[ANON]")
doc_tokenizer.add_tokens("[D]")

model = ColBERT(args.model_name, args.hidden_size, args.compress_size, query_tokenizer, doc_tokenizer).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=NUM_STEPS, min_factor=0.1)

progress_bar = tqdm(range(NUM_STEPS))

steps = 0

while steps < NUM_STEPS:

    for texts, positive_documents, negative_documents, labels in dataloader:

        labels = labels.to(device)
        queries = query_tokenizer(texts, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        queries.attention_mask[:, :] = 1
        queries = queries.to(device)
        pos_docs = doc_tokenizer(positive_documents, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        neg_docs = doc_tokenizer(negative_documents, return_tensors="pt", padding="max_length", max_length=512, truncation=True)

        doc_ids = torch.cat([pos_docs["input_ids"], neg_docs["input_ids"]], dim=0)
        doc_ids = doc_ids.to(device)
        doc_attention_mask = torch.cat([pos_docs["attention_mask"], neg_docs["attention_mask"]], dim=0)
        doc_attention_mask = doc_attention_mask.to(device)

        logits = model(queries["input_ids"], queries["attention_mask"], doc_ids, doc_attention_mask)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        accuracy = (logits.detach().argmax(-1) == 0).float().mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        steps += 1
        progress_bar.update(1)

        progress_bar.set_postfix_str(f"Loss: {loss.item():.2f}, Accuracy: {accuracy.item() * 100}")

        if device == "cuda":
            wandb.log({"stats/loss": loss.item(), "stats/accuracy": accuracy.item()*100})

        if steps >= NUM_STEPS:
            break

torch.save(model.state_dict(), "../models/colbert_wiki_all_mask.bin")
