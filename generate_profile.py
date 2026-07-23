import os
import json
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Generate NL User Profiles using an LLM")
parser.add_argument("--dataset", type=str, default="Amazon/MoviesAndTV", choices=["Amazon/MoviesAndTV", "TripAdvisor"], help="Dataset name")
parser.add_argument("--input_features", type=str, default="", help="Path to input k-best features JSON file")
parser.add_argument("--output", type=str, default="", help="Path to output user profiles JSON file")
parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Hugging Face model ID")
parser.add_argument("--prompt_type", type=str, default="auto", choices=["auto", "amazon", "tripadvisor"], help="Type of prompt template to use")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Resolve default paths if not provided
if not args.input_features:
    if "TripAdvisor" in args.dataset:
        args.input_features = "user_profiles/tripadvisor_k_best_features.json"
    else:
        args.input_features = "user_profiles/amazon_k_best_features.json"

if not args.output:
    if "TripAdvisor" in args.dataset:
        args.output = "user_profiles/trip_advisor_profiles_mistral.json"
    else:
        args.output = "user_profiles/amazon_profiles_mistral.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=dtype,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.pad_token = tokenizer.eos_token

def create_prompt(reviews):
    nl = "\n"
    return f"""Summarize in a single paragraph using the first person my general movie and tv preferences based on my reviews. Do not mention the word reviews.
Reviews:
{nl.join(reviews)}
Summary:"""

def create_trip_advisor_prompt(reviews):
    nl = "\n"
    return f"""Summarize in a single paragraph summary, using the first person, describing my general trip and hotel preferences based on my reviews. Do not mention the word reviews in the summary.
Reviews:
{nl.join(reviews)}
Summary:"""

# Determine prompt template
if args.prompt_type == "tripadvisor" or (args.prompt_type == "auto" and "TripAdvisor" in args.dataset):
    prompt_builder = create_trip_advisor_prompt
else:
    prompt_builder = create_prompt

os.makedirs(os.path.dirname(args.output) or "user_profiles", exist_ok=True)

with open(args.input_features, "r", encoding="utf-8") as f:
    data = json.load(f)

profiles = []
for user in tqdm(data, total=len(data), desc="Generating profiles"):
    user_id = user["user_id"]
    profile_data = {"user_id": user_id}
    
    reviews = []
    for feature in user.get("k_best_data", [])[:5]:
        max_len = 5 if len(feature["reviews"]) > 5 else len(feature["reviews"])
        for review in feature["reviews"][:max_len]:
            reviews.append(f"- {review}")

    prompt = prompt_builder(reviews)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    gen_tokens = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=200,
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])[0]
    summary = gen_text.strip().replace("\n", "").replace("</s>", "")
    profile_data["profile"] = summary
    profiles.append(profile_data)

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=4)

print(f"Saved generated profiles to {args.output}")
