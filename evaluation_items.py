from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, Dataset
import json
import numpy as np
import pandas as pd
import torch
import argparse

from cornac.metrics import NDCG, MAP, RMSE, MAE

parser = argparse.ArgumentParser(description="model configuration")

# Add arguments
parser.add_argument(
    "--pretrained_model",
    type=str,
    required=False,
    default="gpt2",
    help="Pretrained model name",
)

parser.add_argument(
    "--profiles", type=str, required=True, help="profiles"
)

parser.add_argument(
    "--sampling_file", type=str, required=False, help="sampling"
)

parser.add_argument(
    "--output", type=str, required=False, help="output"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="seed"
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load the dataset
data_files = {
    "test": "datasets/Amazon/MoviesAndTV/test.jsonl"
}

user_items_path = "datasets/Amazon/MoviesAndTV/user_items.jsonl"


def load_jsonl_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


user_items = load_jsonl_records(user_items_path)

# Build user review history from user_items
user_history = {}
for _, row in user_items.iterrows():
    user_id = row["user"]
    items = row.get("items", [])
    # store full item dicts so we can filter out the target item at prompt time
    user_history[user_id] = items


dataset = load_dataset("json", data_files=data_files)
model_name = args.pretrained_model
tokenizer = AutoTokenizer.from_pretrained('gpt2', device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# convert input to prompt
def convert_to_prompt(example):
    user_id = example["user"]
    # Filter out the review of the target item to avoid leaking the label
    user_items_list = user_history.get(user_id, [])

    target_item_id = example.get("item")
    target_title = example.get("title")

    history_text = []
    for it in user_items_list:
        # skip if this history entry corresponds to the item being predicted
        if target_item_id is not None and it.get("item") == target_item_id:
            continue
        if target_title is not None and it.get("title") == target_title:
            continue
        t = it.get("title", "")
        d = it.get("description", "")
        r = it.get("review", "")
        if t or d or r:
            history_text.append(f"Title: {t} | Description: {d} | Review: {r}")

    review_history = " | ".join(history_text) if history_text else "No review history available"

    title = example.get("title", "this item")
    description = example.get("description", "No description available")

    example["prompt"] = (
        f"User Review History: {review_history} Based on my review history, "
        f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
        f"i would give \"{title}\"; Description: \"{description}\" a rating of"
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


def compute_scaled_metrics(eval_pred):
    scaled_predictions, scaled_labels = eval_pred
    scaled_predictions = scaled_predictions[:, 0]

    # Inverse scaling: convert normalized [0, 1] values back to original [1, 5] star ratings
    def inverse_scale(values, min_val=1, max_val=5):
        return [s * (max_val - min_val) + min_val for s in values]

    original_predictions = np.array(inverse_scale(scaled_predictions))
    original_labels = np.array(inverse_scale(scaled_labels))

    # Pull user and item metadata vectors directly from the current evaluation slice
    users = np.array(trainer.eval_dataset['user'])
    items = np.array(trainer.eval_dataset['item'])

    # 1. Global Rating Metrics
    rmse_score = RMSE().compute(original_labels, original_predictions)
    mae_score = MAE().compute(original_labels, original_predictions)

    # 2. Per-User Ranking Metrics (Sakai Condensed Lists Strategy)
    map_eval = MAP()
    ndcg10_eval = NDCG(k=10)

    user_map_scores = []
    user_ndcg_scores = []

    unique_users = np.unique(users)
    
    for u in unique_users:
        # Filter indices belonging strictly to the current user
        u_indices = np.where(users == u)[0]
        
        # Sakai (2007) constraint: We only extract and pass items explicitly rated in this test split
        u_items = items[u_indices]
        u_true_ratings = original_labels[u_indices]
        u_pred_ratings = original_predictions[u_indices]

        # Define ground-truth positive items using the 4.0 relevance threshold
        gt_pos = u_items[u_true_ratings >= 4.0]
        
        # If the user has no positive ground-truth items in the test slice, skip them
        if len(gt_pos) == 0:
            continue

        # --- MAP Calculation ---
        # Signature: compute(item_indices, pd_scores, gt_pos)
        u_map = map_eval.compute(
            item_indices=u_items, 
            pd_scores=u_pred_ratings, 
            gt_pos=gt_pos
        )

        # --- NDCG Calculation ---
        # Signature: compute(gt_pos, pd_rank)
        # Create pd_rank by sorting u_items by u_pred_ratings in descending order (-)
        sorted_indices = np.argsort(-u_pred_ratings)
        pd_rank = u_items[sorted_indices]

        u_ndcg = ndcg10_eval.compute(
            gt_pos=gt_pos, 
            pd_rank=pd_rank
        )

        user_map_scores.append(u_map)
        user_ndcg_scores.append(u_ndcg)

    # Compute final averages across all valid evaluation users
    final_map = np.mean(user_map_scores) if user_map_scores else 0.0
    final_ndcg = np.mean(user_ndcg_scores) if user_ndcg_scores else 0.0

    # 3. Save detailed outputs to a JSONL file if specified
    if args.output:
        output = [{
            'user': str(u_id), 
            'item': str(i_id), 
            'predicted_rating': float(p), 
            'true_rating': float(t)
        } for (u_id, i_id, p, t) in zip(users, items, original_predictions, original_labels)]
        
        with open(args.output, 'w') as outfile:
            for entry in output:
                json.dump(entry, outfile)
                outfile.write('\n')

    return {
        "rmse": float(rmse_score), 
        "mae": float(mae_score), 
        "map": float(final_map), 
        "ndcg10": float(final_ndcg)
    }
    
    
trainer = Trainer(
    model=model,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_scaled_metrics,
)
print('saving to:', args.output)
# Evaluate the model
results = trainer.evaluate()
print(results)