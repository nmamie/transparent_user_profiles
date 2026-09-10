"""Behavioural validation of natural-language profile steering.

The UMAP trajectory produced by `user_profile_interpretability.py --run-perturbation`
shows that editing a user profile moves its internal representation. This script tests
the behaviourally meaningful question that movement alone does not answer: does the edit
move the model's *predicted ratings and rankings* toward the edited target genre?

It reports three analyses:

  1. `--run-variance`   Decomposes the variance of predicted ratings over a
                        profile x item grid into item, profile, and interaction
                        components. Genre-selective steering is an interaction effect,
                        so this bounds how much of the model's behaviour it could explain.
  2. `--run-steering`   Scores genre-labelled item pools under each perturbation state and
                        tests for a genre-selective shift in predicted rating and in rank.
                        Rank is invariant to a uniform additive shift in predicted rating
                        and therefore isolates the profile x item interaction.
  3. `--run-sanity`     Predictive sanity check against ground-truth ratings, so effect
                        sizes can be read relative to the model's actual dynamic range.

Note on model choice: pass a checkpoint trained with `--context_out "item title and
description"` for any content-matching analysis. A title-only checkpoint never observes
the genre evidence that a genre-selective effect would have to be computed from, and will
report a null for that reason alone.

Example:
    python behavioral_validation.py \
        --model_path out/amazon-out-reproduce-profile-title-and-description \
        --run-variance --run-steering --run-sanity
"""

import argparse
import json
import os
import random
import re

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Genre pools are built by weak supervision over item descriptions. Keywords are
# genre-diagnostic phrases rather than bare genre names, and an item is assigned to a pool
# only if it matches exactly one pool, to limit cross-genre contamination.
GENRE_KEYWORDS = {
    "romance": ["romantic comedy", "romance", "love story", "falls in love", "love affair", "wedding"],
    "action": ["action-packed", "action packed", "high-octane", "martial arts", "fight scenes",
               "gunfights", "car chase", "explosive action", "kung fu"],
    "crime": ["detective", "murder mystery", "crime thriller", "noir", "heist", "serial killer",
              "investigation", "homicide"],
}

GENRES = ("romance", "action", "crime")

# The three counterfactual perturbation states, expressed at the length of a real training
# profile (median 87 words). Short probe strings fall below the shortest profile seen during
# fine-tuning and put the model out of distribution.
PERTURBATION_STATES = {
    "baseline": (
        "Personally, I tend to enjoy romantic comedies with authentic love stories and lighthearted romance. "
        "I appreciate films that focus on tender relationships, emotional chemistry between the leads, and warm, "
        "hopeful endings that leave me feeling good. I have a soft spot for well-written dialogue between couples "
        "and for stories about people slowly falling in love. I am less drawn to films built around violence or "
        "grim subject matter. Overall, I prefer movies that are heartfelt, charming, and emotionally sincere rather "
        "than dark or cynical in their outlook."
    ),
    "weak": (
        "Personally, I tend to enjoy romantic comedies, but I also like a lot of action scenes and fast-paced thrillers. "
        "I appreciate films that mix warmth and humor with excitement, chases, and a sense of momentum that keeps me "
        "engaged throughout. I have a soft spot for stories where the relationship develops against a backdrop of "
        "danger or high stakes. I am less drawn to films that are slow or purely sentimental. Overall, I prefer movies "
        "that balance emotional connection with energy, tension, and moments of genuine adrenaline."
    ),
    "strong": (
        "Personally, my favorite genre is crime and mystery thrillers with suspenseful plots, detective investigations, "
        "and dark puzzles. I appreciate films that unfold like an intricate case, where clues accumulate and the "
        "resolution genuinely surprises me. I have a soft spot for morally ambiguous characters, gritty urban settings, "
        "and stories about obsession and consequence. I am less drawn to films that are lighthearted or sentimental. "
        "Overall, I prefer movies that are tense, atmospheric, and psychologically complex rather than warm or "
        "reassuring in their outlook."
    ),
}


def resolve_device(device=None):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(model_path, problem_type="regression")
    model.eval().to(device)
    return model, tokenizer


def load_item_descriptions(dataset_dir):
    """Maps item title -> description, mirroring the `item title and description` prompt context."""
    with open(os.path.join(dataset_dir, "item.json"), "r", encoding="utf-8") as f:
        items = json.load(f)
    title_to_desc = {}
    for it in items:
        title = it.get("title")
        if title and title not in title_to_desc:
            title_to_desc[title] = it.get("description") or "No description available"
    return title_to_desc


def build_genre_pools(dataset_dir, n_per_genre, seed):
    """Weak-supervision genre pools: an item joins a pool only if it matches that pool alone."""
    with open(os.path.join(dataset_dir, "item.json"), "r", encoding="utf-8") as f:
        items = json.load(f)

    def matches(item, keywords):
        haystack = f"{item.get('title', '')} {item.get('description', '')}".lower()
        return any(k in haystack for k in keywords)

    pools = {g: set() for g in GENRES}
    for item in items:
        title = item.get("title")
        if not title:
            continue
        hits = [g for g in GENRES if matches(item, GENRE_KEYWORDS[g])]
        if len(hits) == 1:
            pools[hits[0]].add(title)

    rng = random.Random(seed)
    sized = {}
    for g in GENRES:
        titles = sorted(pools[g])
        rng.shuffle(titles)
        sized[g] = titles[:n_per_genre]
    return sized


def build_prompt(profile, title, title_to_desc=None):
    """Reproduces the prompt template used in train.py / evaluate.py."""
    if title_to_desc is None:
        output_ctx = title
    else:
        output_ctx = f"{title}: {title_to_desc.get(title, 'No description available')}"
    return (
        f"Input Context: {profile} Based on the input context, "
        f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
        f"I would give \"{output_ctx}\" a rating of"
    )


@torch.no_grad()
def predict_ratings(model, tokenizer, prompts, device, batch_size=16, max_length=300):
    """Predicted ratings on the original 1-5 scale (training labels were scaled to [0,1])."""
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1)
        out.extend((logits * 4 + 1).cpu().numpy().tolist())
    return np.asarray(out)


def run_sanity(model, tokenizer, args, device, title_to_desc):
    """Predictive sanity check: correlation and error against ground-truth test ratings."""
    print("\n=== Predictive sanity check ===")
    profiles = {p["user_id"]: p["profile"] for p in json.load(open(args.profiles, encoding="utf-8"))}
    with open(os.path.join(args.dataset_dir, "test.jsonl"), encoding="utf-8") as f:
        test = [json.loads(line) for line in f]

    sample = [e for e in test if e["user"] in profiles]
    random.Random(args.seed).shuffle(sample)
    sample = sample[:args.n_sanity]

    prompts = [build_prompt(profiles[e["user"]], e.get("title", "this item"), title_to_desc)
               for e in sample]
    preds = predict_ratings(model, tokenizer, prompts, device, args.batch_size)
    labels = np.asarray([float(e["label"]) for e in sample])

    rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
    print(f"  n={len(sample)}")
    print(f"  true   : mean={labels.mean():.3f} std={labels.std():.3f}")
    print(f"  predict: mean={preds.mean():.3f} std={preds.std():.3f} "
          f"[min={preds.min():.3f} max={preds.max():.3f}]")
    print(f"  RMSE={rmse:.3f}  MAE={np.abs(preds - labels).mean():.3f}  "
          f"Pearson r={np.corrcoef(preds, labels)[0, 1]:.4f}")
    print("  -> effect sizes below should be read relative to the prediction std above")


def run_variance(model, tokenizer, args, device, title_to_desc, pools):
    """Decomposes predicted-rating variance into item, profile, and interaction components.

    Genre matching is an interaction effect: if the interaction term is small, profile edits
    can only move all candidates together, regardless of their content.
    """
    print("\n=== Variance decomposition (profile x item grid) ===")
    all_profiles = json.load(open(args.profiles, encoding="utf-8"))
    rng = random.Random(args.seed)
    grid_profiles = [p["profile"] for p in rng.sample(all_profiles, args.n_grid_profiles)]

    grid_items = []
    for g in GENRES:
        grid_items.extend(pools[g][:args.n_grid_items_per_genre])

    matrix = np.zeros((len(grid_profiles), len(grid_items)))
    for i, profile in enumerate(grid_profiles):
        prompts = [build_prompt(profile, t, title_to_desc) for t in grid_items]
        matrix[i, :] = predict_ratings(model, tokenizer, prompts, device, args.batch_size)

    grand = matrix.mean()
    ss_total = ((matrix - grand) ** 2).sum()
    ss_item = matrix.shape[0] * ((matrix.mean(axis=0) - grand) ** 2).sum()
    ss_profile = matrix.shape[1] * ((matrix.mean(axis=1) - grand) ** 2).sum()
    ss_interaction = ss_total - ss_item - ss_profile

    print(f"  grid: {len(grid_profiles)} real profiles x {len(grid_items)} items")
    print(f"  predicted rating: mean={matrix.mean():.3f} std={matrix.std():.3f}")
    print(f"  item identity : {100 * ss_item / ss_total:.1f}%")
    print(f"  user profile  : {100 * ss_profile / ss_total:.1f}%")
    print(f"  interaction   : {100 * ss_interaction / ss_total:.1f}%  <- ceiling for content matching")
    return {
        "item_pct": 100 * ss_item / ss_total,
        "profile_pct": 100 * ss_profile / ss_total,
        "interaction_pct": 100 * ss_interaction / ss_total,
    }


def run_steering(model, tokenizer, args, device, title_to_desc, pools):
    """Tests whether a profile edit shifts predictions toward the edited target genre.

    Pre-specified hypotheses:
      H1  the predicted-rating change for crime items exceeds that for romance and action
          items under the crime-targeting edit (state 3);
      H2  crime items move up the predicted ranking from state 1 to state 3. Ranking is
          invariant to a uniform additive shift and so isolates the interaction.
    """
    print("\n=== Genre-selective steering test ===")
    ratings = {g: {} for g in GENRES}
    for g in GENRES:
        for state, profile in PERTURBATION_STATES.items():
            prompts = [build_prompt(profile, t, title_to_desc) for t in pools[g]]
            ratings[g][state] = predict_ratings(model, tokenizer, prompts, device, args.batch_size)
        print(f"  [scored] {g}: {len(pools[g])} items")

    print(f"\n  mean predicted rating (N={args.n_per_genre}/genre)")
    for g in GENRES:
        print(f"    {g:8s}: " + ", ".join(f"{s}={ratings[g][s].mean():.3f}" for s in PERTURBATION_STATES))

    print("\n  MAIN EFFECT (all genres pooled)")
    for state in ("weak", "strong"):
        deltas = np.concatenate([ratings[g][state] - ratings[g]["baseline"] for g in GENRES])
        t, p = stats.ttest_1samp(deltas, 0.0)
        print(f"    {state:6s}: mean_delta={deltas.mean():+.4f}  t={t:.3f}  p={p:.4g}")

    print("\n  H1 -- genre-selectivity of the rating change")
    for state in ("weak", "strong"):
        groups = [ratings[g][state] - ratings[g]["baseline"] for g in GENRES]
        f_stat, p = stats.f_oneway(*groups)
        print(f"    {state:6s}: " + ", ".join(f"{g}={d.mean():+.4f}" for g, d in zip(GENRES, groups)))
        print(f"            ANOVA F={f_stat:.3f} p={p:.4g}")
        d_crime = ratings["crime"][state] - ratings["crime"]["baseline"]
        for other in ("romance", "action"):
            d_other = ratings[other][state] - ratings[other]["baseline"]
            t, p_two = stats.ttest_ind(d_crime, d_other, equal_var=False)
            one_sided = p_two / 2 if t > 0 else 1 - p_two / 2
            print(f"            crime > {other:8s}: Welch t={t:.3f} one-sided p={one_sided:.4g}")

    # H2: ranking within the pooled candidate set
    genre_of = np.concatenate([np.full(len(pools[g]), g) for g in GENRES])
    scores = {s: np.concatenate([ratings[g][s] for g in GENRES]) for s in PERTURBATION_STATES}
    n_items = len(genre_of)
    ranks = {}
    for state, sc in scores.items():
        order = np.argsort(-sc)
        r = np.empty(n_items)
        r[order] = np.arange(1, n_items + 1)
        ranks[state] = r

    print(f"\n  H2 -- ranking within pooled candidate set ({n_items} items, lower rank = better)")
    for g in GENRES:
        m = genre_of == g
        print(f"    {g:8s} mean rank: " + ", ".join(f"{s}={ranks[s][m].mean():.1f}" for s in PERTURBATION_STATES))

    for state in ("weak", "strong"):
        print(f"\n    rank change vs baseline ({state}; negative = moved up)")
        groups = []
        for g in GENRES:
            m = genre_of == g
            d = ranks[state][m] - ranks["baseline"][m]
            groups.append(d)
            t, p = stats.ttest_1samp(d, 0.0)
            print(f"      {g:8s}: {d.mean():+.2f} ranks  (t={t:.3f} p={p:.4g})")
        f_stat, p = stats.f_oneway(*groups)
        print(f"      ANOVA F={f_stat:.3f} p={p:.4g}")
        t, p_two = stats.ttest_ind(groups[2], groups[0], equal_var=False)
        one_sided = p_two / 2 if t < 0 else 1 - p_two / 2
        print(f"      crime moved up more than romance? Welch t={t:.3f} one-sided p={one_sided:.4g}")

    print(f"\n  genre share of top-{args.top_k} predicted items")
    for state in PERTURBATION_STATES:
        top = np.argsort(-scores[state])[:args.top_k]
        print(f"    {state:8s}: " + ", ".join(
            f"{g}={100 * (genre_of[top] == g).mean():.0f}%" for g in GENRES))

    return {g: {s: v.tolist() for s, v in d.items()} for g, d in ratings.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str,
                        default="out/amazon-out-reproduce-profile-title-and-description",
                        help="fine-tuned checkpoint; use a title-and-description model for content matching")
    parser.add_argument("--dataset_dir", type=str, default="datasets/Amazon/MoviesAndTV")
    parser.add_argument("--profiles", type=str, default="user_profiles/amazon_profiles.json")
    parser.add_argument("--item_context", type=str, default="item title and description",
                        choices=["item title", "item title and description"],
                        help="must match the checkpoint's training context_out")
    parser.add_argument("--n_per_genre", type=int, default=110)
    parser.add_argument("--n_grid_profiles", type=int, default=25)
    parser.add_argument("--n_grid_items_per_genre", type=int, default=8)
    parser.add_argument("--n_sanity", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/behavioral_validation.json")
    parser.add_argument("--run-variance", dest="run_variance", action="store_true")
    parser.add_argument("--run-steering", dest="run_steering", action="store_true")
    parser.add_argument("--run-sanity", dest="run_sanity", action="store_true")
    args = parser.parse_args()

    if not (args.run_variance or args.run_steering or args.run_sanity):
        args.run_variance = args.run_steering = args.run_sanity = True

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[Validation] model={args.model_path}")
    print(f"[Validation] device={device}  item_context='{args.item_context}'")

    model, tokenizer = load_model(args.model_path, device)
    title_to_desc = (load_item_descriptions(args.dataset_dir)
                     if args.item_context == "item title and description" else None)
    pools = build_genre_pools(args.dataset_dir, args.n_per_genre, args.seed)
    print("[Validation] genre pools: " + ", ".join(f"{g}={len(pools[g])}" for g in GENRES))

    results = {"model_path": args.model_path, "item_context": args.item_context, "seed": args.seed}

    if args.run_sanity:
        run_sanity(model, tokenizer, args, device, title_to_desc)
    if args.run_variance:
        results["variance"] = run_variance(model, tokenizer, args, device, title_to_desc, pools)
    if args.run_steering:
        results["ratings"] = run_steering(model, tokenizer, args, device, title_to_desc, pools)
        results["pools"] = pools

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Validation] saved results to {args.output}")


if __name__ == "__main__":
    main()
