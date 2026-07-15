from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
import json
import numpy as np
import pandas as pd
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="model configuration")

# Add arguments
parser.add_argument("--output_dir", type=str, required=True, help="output name inside out/")
parser.add_argument(
    "--pretrained_model",
    type=str,
    required=False,
    default="gpt2",
    help="Pretrained model name",
)

parser.add_argument(
    "--context_in", type=str, required=False, default="user profile", choices=["user profile", "review history", "item-review history"], help="input context for the prompt"
)

parser.add_argument(
    "--context_out", type=str, required=False, default="item title", choices=["item title", "item title and description"], help="output context for the prompt"
)

parser.add_argument(
    "--num_train_epochs", type=int, required=False, default=5, help="num train epochs"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="num train epochs"
)
parser.add_argument(
    "--lr", type=float, required=False, default=3e-4, help="learning rate"
)
parser.add_argument(
    "--batch_size", type=int, required=False, default=32, help="batch size"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Load the dataset (for demonstration, we'll use the IMDb dataset)
# Load the dataset
data_files = {
    "train": "datasets/Amazon/MoviesAndTV/train.jsonl",
    "eval": "datasets/Amazon/MoviesAndTV/validation.jsonl",
    # "train": "datasets/TripAdvisor/train.jsonl",
    # "test": "datasets/TripAdvisor/test.jsonl",
}

with open("user_profiles/amazon_profiles.json") as f:
    profiles_data = json.load(f)

profiles = {}

for i in profiles_data:
    user_id = i["user_id"]
    user_profile = i["profile"]
    profiles[user_id] = user_profile


dataset = load_dataset("json", data_files=data_files)

# Load user items review history if needed
user_reviews_map = {}
item_reviews_map = {}
if args.context_in in ["review history", "item-review history"]:
    import os
    train_file = data_files["train"]
    parent_dir = os.path.dirname(train_file)
    user_items_path = os.path.join(parent_dir, "user_items.jsonl")
    if os.path.exists(user_items_path):
        with open(user_items_path, "r", encoding="utf-8") as f:
            user_items_data = json.load(f)
        for u_id, items in user_items_data.items():
            user_reviews_map[u_id] = items
            for it in items:
                it_id = it.get("item_id")
                if it_id:
                    if it_id not in item_reviews_map:
                        item_reviews_map[it_id] = []
                    item_reviews_map[it_id].append({
                        "user_id": u_id,
                        "review": it.get("review", ""),
                        "rating": it.get("rating"),
                        "title": it.get("title", ""),
                        "description": it.get("description", "")
                    })

model_name = args.pretrained_model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def convert_to_prompt(example):
    user_id = example["user"]
    # Build input context based on the selected option
    def summarize_reviews(reviews, max_reviews=5):
        if not reviews:
            return "No review history available"
        # reviews may be list of dicts or list of strings
        out = []
        for r in reviews[-max_reviews:]:
            if isinstance(r, dict):
                text = r.get("text") or r.get("review") or r.get("body") or r.get("comment")
                rating = r.get("rating") or r.get("score")
                if rating is not None:
                    out.append(f"[{rating}] {text}" if text else f"[{rating}]")
                else:
                    out.append(text if text else "<no-text>")
            else:
                out.append(str(r))
        return " || ".join([o for o in out if o])

    if args.context_in == "user profile":
        # profile entry may contain summary or review history inside
        profile = profiles.get(user_id)
        if isinstance(profile, dict):
            # prefer explicit profile text, otherwise fall back to reviews inside profile
            input_context = profile.get("summary") or profile.get("profile") or summarize_reviews(profile.get("review_history") or profile.get("reviews"))
        else:
            input_context = profile or "No profile available"
    elif args.context_in == "review history":
        target_item = example.get("item")
        user_revs = user_reviews_map.get(user_id, [])
        filtered_revs = [r for r in user_revs if r.get("item_id") != target_item]
        if filtered_revs:
            input_context = summarize_reviews(filtered_revs)
        else:
            # try several possible keys in the example and in the stored profile
            input_context = example.get("review_history") or example.get("reviews") or example.get("user_reviews")
            if not input_context:
                # fallback to any reviews inside the profile data
                profile = profiles.get(user_id, {})
                if isinstance(profile, dict):
                    input_context = profile.get("review_history") or profile.get("reviews")
                else:
                    input_context = None
            input_context = summarize_reviews(input_context)
    elif args.context_in == "item-review history":
        target_item = example.get("item")
        item_revs = item_reviews_map.get(target_item, [])
        filtered_revs = [r for r in item_revs if r.get("user_id") != user_id]
        if filtered_revs:
            input_context = summarize_reviews(filtered_revs)
        else:
            # item-review history: reviews for this specific item
            # try example keys first
            item_reviews = example.get("item_review_history") or example.get("item_reviews") or example.get("reviews_for_item")
            if not item_reviews:
                # sometimes dataset uses a nested item object
                item = example.get("item") or {}
                item_reviews = item.get("reviews") if isinstance(item, dict) else None
            if not item_reviews:
                # fallback: look for reviews that mention the item id in profile
                profile = profiles.get(user_id, {})
                if isinstance(profile, dict):
                    item_reviews = profile.get("item_review_history") or profile.get("item_reviews")
                else:
                    item_reviews = None
            input_context = summarize_reviews(item_reviews)
    else:
        input_context = ""

    # Build output context
    title = example.get("title", "this item")
    if args.context_out == "item title":
        output_ctx = title
    else:
        desc = example.get("description", "No description available")
        output_ctx = f"{title}: {desc}"

    example["prompt"] = (
        f"Input Context: {input_context} Based on the input context, "
        f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
        f"I would give \"{output_ctx}\" a rating of"
    )
    return example


dataset = dataset.map(convert_to_prompt)


# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["prompt"], truncation=True, padding="max_length", max_length=300
    )
    # Scale the labels from [1,5] to [0,1]
    min_val, max_val = 1, 5
    scaled_labels = [
        (label - min_val) / (max_val - min_val) for label in examples["label"]
    ]

    tokenized_output["label"] = scaled_labels
    return tokenized_output


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, device_map="auto"
)
model.config.pad_token_id = model.config.eos_token_id


def compute_metrics(eval_pred):
    scaled_predictions, scaled_labels = eval_pred

    # Assuming this is a regression problem, we'll take the first value from the output logits
    scaled_predictions = scaled_predictions[:, 0]

    # Inverse scaling
    def inverse_scale(values, min_val=1, max_val=5):
        return [s * (max_val - min_val) + min_val for s in values]

    original_predictions = inverse_scale(scaled_predictions)
    original_labels = inverse_scale(scaled_labels)

    # Compute the metrics
    rmse = np.sqrt(
        ((np.array(original_predictions) - np.array(original_labels)) ** 2).mean()
    )
    mae = np.abs(np.array(original_predictions) - np.array(original_labels)).mean()

    return {"rmse": rmse, "mae": mae}

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement after which training will be stopped.
    early_stopping_threshold=0.0  # Minimum improvement to qualify as an improvement.
)

results_dir = Path("out") / args.output_dir
results_dir.mkdir(parents=True, exist_ok=True)

# Define training arguments and set up Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    logging_dir=str(results_dir / "logs"),
    logging_steps=1000,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    learning_rate=args.lr,
    num_train_epochs=args.num_train_epochs,
    output_dir=str(results_dir),
    remove_unused_columns=True,  # Important!
    seed=args.seed,
    lr_scheduler_type="linear",
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# Train the model
trainer.train()
trainer.save_model(str(results_dir))
tokenizer.save_pretrained(str(results_dir))
