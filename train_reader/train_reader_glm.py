import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import build_inputs_for_generation, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import wandb
import argparse
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--texts_to_re_id", required=True, type=str, help="A JSONL file that contains the anonymized local context, the retrieved chunks, and the targets.")
    parser.add_argument("--output_path", required=True, type=Path, help="Path where to save the outputs.")
    parser.add_argument("--seed", default=782364, type=int, help="Random number generator seed.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size to pass to the ranker.")
    parser.add_argument("--num_steps", default=5000, type=int, help="Number of training steps.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="Maximum learning rate during training.")
    parser.add_argument("--gradient_accumulation", default=32, type=int, help="Number of gradient accumulation steps to do (train step batch size is therefore batch_size*gradient_accumulation).")
    parser.add_argument("--warmup_ratio", default=0.05, type=float, help="The percentage of training steps to use as warmup for the learning rate.")

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
    retrieved_chunks = [datapoint[1] for datapoint in data]
    targets = [datapoint[2] for datapoint in data]
    return (texts, retrieved_chunks, targets)


if __name__ == "__main__":
    args = parse_arguments()

    NUM_STEPS = args.num_steps
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    gradient_accumulation = args.gradient_accumulation

    dataset = Dataset(args.texts_to_re_id)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-roberta-large", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-roberta-large", trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * NUM_STEPS), num_training_steps=NUM_STEPS, min_factor=0.1)

    progress_bar = tqdm(range(max(NUM_STEPS, len(dataloader)//gradient_accumulation)))

    if device == "cuda":
        wandb.init(
            name="GLM_ColBERT_Retrieval",
            project="PROJECT",
            entity="USER",
            allow_val_change=True,
            reinit=True,
        )

    steps, iterations, total_loss, total_accuracy, total_exact_match, total_first_accuracy = 0, 0, 0, 0, 0, 0

    columns = ["Text", "Retrieved", "Predictions", "Labels", "Targets"]
    pred_vs_lab = wandb.Table(columns=columns)

    tokenizer.truncation_side = "right"

    while steps < NUM_STEPS:

        for texts, retrieved_chunks, targets in dataloader:
            inputs = tokenizer(texts, text_pair=retrieved_chunks, return_tensors="pt", padding=True, max_length=512, truncation=True)
            inputs = inputs.to(device)
            gen_inputs = build_inputs_for_generation(tokenizer, inputs, max_gen_length=512, targets=targets, padding=True)
            gen_inputs = gen_inputs.to(device)
            outputs = model(**gen_inputs)
            loss = outputs.loss

            loss = loss / gradient_accumulation
            loss.backward()

            gold_labels = gen_inputs["labels"]
            logits = outputs.logits.detach().argmax(-1)

            gold_labels_first = []
            logits_first = []

            for gl, l in zip(gold_labels, logits):
                for i, t in enumerate(gl):
                    if t.item() != -100:
                        gold_labels_first.append(t.item())
                        logits_first.append(l[i].item())
                        break

            gold_labels_first = torch.tensor(gold_labels_first)
            logits_first = torch.tensor(logits_first)

            logits = logits[gold_labels != -100]
            gold_labels = gold_labels[gold_labels != -100]

            pred_vs_lab.add_data(texts[0], retrieved_chunks[0], tokenizer.decode(logits), tokenizer.decode(gold_labels), targets[0])

            accuracy = (logits == gold_labels).float().mean() / gradient_accumulation
            first_accuracy = (logits_first == gold_labels_first).float().mean() / gradient_accumulation
            exact_match = float(tokenizer.decode(logits) == tokenizer.decode(gold_labels)) / gradient_accumulation

            total_accuracy += accuracy
            total_first_accuracy += first_accuracy
            total_exact_match += exact_match
            total_loss += loss.item()

            iterations += 1
            if iterations == gradient_accumulation:

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
                progress_bar.update(1)

                progress_bar.set_postfix_str(f"Loss: {total_loss:.2f}, Accuracy: {total_accuracy*100:.2f}")

                if device == "cuda":
                    wandb.log({"stats/loss": total_loss, "stats/accuracy": total_accuracy*100, "stats/first_accuracy": total_first_accuracy*100, "stats/exact_match": total_exact_match*100, "examples": pred_vs_lab})

                if steps > NUM_STEPS:
                    break

                pred_vs_lab = wandb.Table(columns=columns)

                iterations, total_loss, total_accuracy, total_first_accuracy, total_exact_match = 0, 0, 0, 0, 0

        optimizer.zero_grad()
        iterations = 0
        # steps = NUM_STEPS
        if device == "cuda":
            wandb.log({"examples": pred_vs_lab})

    torch.save(model.state_dict(), args.output_path)
