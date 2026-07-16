from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset
import json
import numpy as np
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
    "--sampling_file", type=str, required=False, help="sampling"
)
parser.add_argument(
    "--profiles", type=str, required=True, help="profiles"
)
parser.add_argument(
    "--output", type=str, required=False, help="output"
)
parser.add_argument(
    "--add_profile", type=str, required=False, help="add to profile"
)
parser.add_argument(
    "--context_in", type=str, required=False, default="user profile", choices=["user profile", "review history", "item-review history"], help="input context for the prompt"
)
parser.add_argument(
    "--context_out", type=str, required=False, default="item title", choices=["item title", "item title and description"], help="output context for the prompt"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="seed"
)
parser.add_argument(
    "--summary_file", type=str, required=False, default="results/evaluation_summary.json", help="path to save aggregated evaluation metrics"
)

args = parser.parse_args()

from utils import set_random_seeds
set_random_seeds(args.seed)

# Load the dataset
data_files = {
    # "test": "datasets/TripAdvisor/test.jsonl",
    "test": "datasets/Amazon/MoviesAndTV/test.jsonl"
}

with open(args.profiles) as f:
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
    test_file = data_files["test"]
    parent_dir = os.path.dirname(test_file)
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

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# convert input to prompt
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

if args.summary_file:
    import os
    results_to_save = {
        "model_name": model_name,
        "context_in": args.context_in,
        "context_out": args.context_out,
        "rmse": results.get("eval_rmse") or results.get("rmse"),
        "mae": results.get("eval_mae") or results.get("mae"),
        "map": results.get("eval_map") or results.get("map"),
        "ndcg10": results.get("eval_ndcg10") or results.get("ndcg10")
    }
    
    summary_data = []
    if os.path.exists(args.summary_file):
        try:
            with open(args.summary_file, "r") as sf:
                summary_data = json.load(sf)
        except Exception:
            summary_data = []
            
    updated = False
    for entry in summary_data:
        if (entry.get("model_name") == model_name and
            entry.get("context_in") == args.context_in and 
            entry.get("context_out") == args.context_out):
            entry.update(results_to_save)
            updated = True
            break
    if not updated:
        summary_data.append(results_to_save)
        
    os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
    with open(args.summary_file, "w") as sf:
        json.dump(summary_data, sf, indent=2)
    print(f"Metrics saved to {args.summary_file}")