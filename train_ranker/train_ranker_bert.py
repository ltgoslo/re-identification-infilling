from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from dataset import InputDataset, TrainDataset, NegativeReducedDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
from metrics import top_k_accuracy


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open("../datasets/train_texts.pkl", "rb") as f:
        articles = pickle.load(f)

    with open("../datasets/train_names.pkl", "rb") as f:
        labels = pickle.load(f)

    print(len(labels))

    train_articles, dev_articles, train_labels, dev_labels = train_test_split(articles, labels, test_size=.1, random_state=35)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # m_tokenizer = MaskerTokenizer("en_core_web_md", tokenizer, for_loops=True, max_length=128)

    BATCH_SIZE = 32
    DEV_BATCH_SIZE = 512

    train_dataset = TrainDataset(train_articles, train_labels)
    dev_dataset_pos = InputDataset(dev_articles, dev_labels, percentage_true_labels=1.0)
    dev_dataset_neg = NegativeReducedDataset(dev_articles, dev_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dev_dataloader_pos = DataLoader(dev_dataset_pos, batch_size=DEV_BATCH_SIZE)
    dev_dataloader_neg = DataLoader(dev_dataset_neg, batch_size=DEV_BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)

    LR = 3e-6
    EPOCHS = 40
    criterion = nn.MarginRankingLoss(margin=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_steps = EPOCHS * len(train_dataloader)
    warm_steps = num_steps // 10
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-80, end_factor=1, total_iters=warm_steps)
    decayer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps - warm_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, decayer], [warm_steps])

    wandb.init(
        project="PROJECT",
        entity="USER",
        config={
            "model": "BERT",
            "Dataset": "TAB",
            "Loss": "Ranking",
            "lr": LR,
            "epochs": EPOCHS,
        }
    )

    progress_bar = tqdm(total=EPOCHS * len(train_dataloader))
    model.to(device)
    step = 0

    for epoch in range(EPOCHS):
        model.train()
        for text, true_labels, false_labels in train_dataloader:
            optimizer.zero_grad()
            true_tokens = tokenizer(text, text_pair=true_labels, truncation=True, padding=True, return_tensors="pt")
            false_tokens = tokenizer(text, text_pair=false_labels, truncation=True, padding=True, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16):
                true_logits = model(**true_tokens.to(device)).logits
                false_logits = model(**false_tokens.to(device)).logits
                loss = criterion(true_logits, false_logits, torch.ones(BATCH_SIZE, 1, dtype=torch.int).to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.update()
            wandb.log({"train/loss": loss.item()}, step=step)
            step += 1

        scores = defaultdict(list)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dev_dataloader_neg):
                tokens = tokenizer(batch[0], text_pair=batch[1], truncation=True, padding=True, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**tokens.to(device)).logits
                for j, text in enumerate(batch[0]):
                    scores[text].append((logits[j].item(), "negative", batch[1][j]))

            for batch, labels in dev_dataloader_pos:
                tokens = tokenizer(batch[0], text_pair=batch[1], truncation=True, padding=True, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**tokens.to(device)).logits
                for j, text in enumerate(batch[0]):
                    scores[text].append((logits[j].item(), "positive", batch[1][j]))

        top_1_acc = top_k_accuracy(scores, k=1)
        top_5_acc = top_k_accuracy(scores, k=5)
        top_10_acc = top_k_accuracy(scores, k=10)

        print(f"Development top-1 accuracy: {top_1_acc}; top-5 accuracy: {top_5_acc}; top-10 accuracy: {top_10_acc}")

        wandb.log({"dev/top-1 accuracy": top_1_acc, "dev/top-5 accuracy": top_5_acc, "dev/top-10 accuracy": top_10_acc})

    wandb.finish()
    torch.save(model.state_dict(), "../models/ranker_model.pt")
