"""Controls for the global profile-attribution analysis.

`user_profile_interpretability.py --run-global-attribution` shows that gradient attribution
over user-profile tokens concentrates on genre words. On its own that observation cannot
distinguish an effect of fine-tuning from a property of the architecture, of the pretrained
backbone, or of the profile corpus itself. This script runs the missing controls:

  finetuned            the fine-tuned regression checkpoint (the condition reported in the paper)
  pretrained_backbone  pretrained GPT-2 with a randomly initialised regression head
                       -> isolates what fine-tuning contributes over the backbone
  random_init          the same architecture with no pretrained weights at all
                       -> isolates what the architecture alone contributes
  finetuned_shuffled   the fine-tuned checkpoint on word-shuffled profiles
                       -> preserves token frequency while destroying word order and meaning

and compares all of them against a frequency baseline (corpus term frequency), which asks
whether high-attribution words are simply frequent words.

Because gradient magnitudes are not comparable across differently-scaled models, the headline
metric is scale-invariant: the ratio of mean attribution on a fixed a-priori genre lexicon to
mean attribution on all other profile words. A ratio near 1.0 means genre words are not
distinguished from other words in that condition.

Example:
    python attribution_controls.py --max_examples 60
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from scipy import stats
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from nnsight import LanguageModel

# Fixed a-priori genre lexicon, defined before running any condition.
GENRE_LEXICON = {
    "romance", "romantic", "comedy", "comedies", "action", "thriller", "thrillers",
    "crime", "mystery", "mysteries", "drama", "dramas", "horror", "western", "westerns",
    "musical", "musicals", "documentary", "documentaries", "animated", "animation",
    "fantasy", "adventure", "suspense", "suspenseful", "noir", "detective", "sci",
}

CONDITIONS = ("finetuned", "pretrained_backbone", "random_init", "finetuned_shuffled")


def resolve_device(device=None):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(condition, finetuned_path, base_model, device, tokenizer, seed):
    """Builds the scoring model for a condition.

    All conditions share the fine-tuned checkpoint's tokenizer so that tokenisation, and
    therefore the set of attributable words, is identical across conditions.
    """
    if condition in ("finetuned", "finetuned_shuffled"):
        instance = AutoModelForSequenceClassification.from_pretrained(
            finetuned_path, problem_type="regression")
    elif condition == "pretrained_backbone":
        # pretrained transformer weights, fresh regression head
        torch.manual_seed(seed)
        instance = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=1, problem_type="regression")
    elif condition == "random_init":
        # same architecture, no pretrained weights
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(base_model, num_labels=1, problem_type="regression")
        instance = AutoModelForSequenceClassification.from_config(config)
    else:
        raise ValueError(f"unknown condition: {condition}")

    if instance.config.pad_token_id is None:
        instance.config.pad_token_id = tokenizer.pad_token_id
    instance.eval()
    return LanguageModel(instance, device_map=device, tokenizer=tokenizer)


def shuffle_words(text, rng):
    """Shuffles word order, preserving the multiset of tokens (and hence term frequency)."""
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def compute_word_attribution(model, tokenizer, profiles, item_titles, max_examples):
    """Mean per-word gradient attribution over profile tokens.

    Mirrors `compute_global_attribution` in user_profile_interpretability.py: attribution is
    the L2 norm of the input-embedding gradient of the predicted rating, summed over the
    subword tokens of a word within a profile, then averaged across profiles containing it.
    """
    totals, counts = Counter(), Counter()
    n = min(len(profiles), max_examples)

    for idx in range(n):
        profile = profiles[idx]
        item_title = item_titles[idx]
        prompt = (
            f"Input Context: {profile} Based on the input context, "
            f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
            f"I would give \"{item_title}\" a rating of"
        )
        toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
        encoding = toks.encodings[0]

        try:
            with model.trace(toks):
                embeddings = model.transformer.wte.output
                embeddings.retain_grad()
                pred = model.output.logits[0]
                pred.backward()
                emb_grad = embeddings.grad.save()
            attr = torch.norm(emb_grad, dim=-1)[0].detach().cpu().numpy()
        except Exception as exc:  # a condition that fails to trace must not silently pass
            raise RuntimeError(f"attribution trace failed on example {idx}: {exc}") from exc

        p_start = prompt.find(profile)
        p_end = p_start + len(profile)

        per_word = {}
        for i, a in enumerate(attr):
            span = encoding.token_to_chars(i)
            if span is None:
                continue
            if p_start == -1 or not (max(span[0], p_start) < min(span[1], p_end)):
                continue
            # expand the subword span to the surrounding full word
            cs, ce = span
            while cs > 0 and prompt[cs - 1].isalpha():
                cs -= 1
            while ce < len(prompt) and prompt[ce].isalpha():
                ce += 1
            word = prompt[cs:ce].strip().lower()
            if re.match(r"^[a-z]{3,}$", word):
                per_word[word] = per_word.get(word, 0.0) + float(a)

        for word, value in per_word.items():
            totals[word] += value
            counts[word] += 1

    return {w: totals[w] / counts[w] for w in totals}, counts


def genre_ratio(word_attr, min_count=2, counts=None):
    """Scale-invariant summary: mean attribution on genre words / mean on all other words."""
    if counts is not None:
        word_attr = {w: v for w, v in word_attr.items() if counts[w] >= min_count}
    genre = [v for w, v in word_attr.items() if w in GENRE_LEXICON]
    other = [v for w, v in word_attr.items() if w not in GENRE_LEXICON]
    if not genre or not other:
        return None
    t, p = stats.ttest_ind(genre, other, equal_var=False)
    return {
        "n_genre_words": len(genre),
        "n_other_words": len(other),
        "mean_genre": float(np.mean(genre)),
        "mean_other": float(np.mean(other)),
        "ratio": float(np.mean(genre) / np.mean(other)),
        "welch_t": float(t),
        "p": float(p),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str,
                        default="out/amazon-out-reproduce-profile-title",
                        help="fine-tuned checkpoint reported in the paper")
    parser.add_argument("--base_model", type=str, default="gpt2",
                        help="backbone for the untrained controls")
    parser.add_argument("--profiles", type=str, default="user_profiles/amazon_profiles.json")
    parser.add_argument("--user_items", type=str,
                        default="datasets/Amazon/MoviesAndTV/user_items.jsonl")
    parser.add_argument("--max_examples", type=int, default=60)
    parser.add_argument("--min_word_count", type=int, default=2,
                        help="minimum number of profiles a word must appear in")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/attribution_controls.json")
    args = parser.parse_args()

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    print(f"[Controls] device={device}  finetuned={args.model_path}  base={args.base_model}")

    profiles_data = json.load(open(args.profiles, encoding="utf-8"))
    rng.shuffle(profiles_data)
    profiles_data = profiles_data[:args.max_examples]
    profiles = [p["profile"] for p in profiles_data]
    user_ids = [p["user_id"] for p in profiles_data]

    # per-user item title, matching compute_global_attribution's behaviour
    user_items = {}
    if os.path.exists(args.user_items):
        user_items = json.load(open(args.user_items, encoding="utf-8"))
    item_titles = []
    for uid in user_ids:
        title = "Tarzan (Walt Disney) [VHS]"
        entries = user_items.get(uid)
        if entries and entries[0].get("title"):
            title = entries[0]["title"]
        item_titles.append(title)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    shuffled_profiles = [shuffle_words(p, rng) for p in profiles]

    results = {}
    for condition in CONDITIONS:
        print(f"\n[Controls] === {condition} ===")
        model = build_model(condition, args.model_path, args.base_model, device, tokenizer, args.seed)
        texts = shuffled_profiles if condition == "finetuned_shuffled" else profiles
        word_attr, counts = compute_word_attribution(
            model, tokenizer, texts, item_titles, args.max_examples)
        filtered = {w: v for w, v in word_attr.items() if counts[w] >= args.min_word_count}
        summary = genre_ratio(word_attr, args.min_word_count, counts)
        results[condition] = {"word_attr": filtered, "genre_summary": summary}

        print(f"  attributable words (>= {args.min_word_count} profiles): {len(filtered)}")
        if summary:
            print(f"  genre-word ratio = {summary['ratio']:.3f}  "
                  f"(genre {summary['mean_genre']:.4f} vs other {summary['mean_other']:.4f}, "
                  f"Welch t={summary['welch_t']:.2f}, p={summary['p']:.3g})")
        top = sorted(filtered.items(), key=lambda kv: -kv[1])[:args.top_k]
        print(f"  top-{args.top_k}: " + ", ".join(
            f"{w}{'*' if w in GENRE_LEXICON else ''}" for w, _ in top))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- frequency baseline ----
    print("\n[Controls] === frequency baseline ===")
    freq = Counter()
    for p in profiles:
        for w in set(re.findall(r"[a-z]{3,}", p.lower())):
            freq[w] += 1
    ft_attr = results["finetuned"]["word_attr"]
    common = [w for w in ft_attr if w in freq]
    rho, p_rho = stats.spearmanr([ft_attr[w] for w in common], [freq[w] for w in common])
    print(f"  Spearman(attribution, corpus frequency) = {rho:.3f}  p={p_rho:.3g}  (n={len(common)})")
    top_freq = [w for w, _ in freq.most_common(args.top_k)]
    top_ft = [w for w, _ in sorted(ft_attr.items(), key=lambda kv: -kv[1])[:args.top_k]]
    print(f"  top-{args.top_k} overlap with most frequent words: "
          f"{len(set(top_ft) & set(top_freq))}/{args.top_k}")
    # The paper's attribution sums saliency over a word's subword tokens, so words that split
    # into more tokens accumulate more of it regardless of meaning. Quantify that confound.
    ntok = [len(tokenizer(" " + w)["input_ids"]) for w in common]
    rho_tok, p_tok = stats.spearmanr([ft_attr[w] for w in common], ntok)
    print(f"  Spearman(attribution, subword count)    = {rho_tok:+.3f}  p={p_tok:.3g}")
    results["frequency_baseline"] = {
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "top_overlap": len(set(top_ft) & set(top_freq)),
        "spearman_subword_rho": float(rho_tok), "spearman_subword_p": float(p_tok),
    }

    # ---- cross-condition comparison against the fine-tuned condition ----
    print("\n[Controls] === agreement with the fine-tuned condition ===")
    comparisons = {}
    for condition in CONDITIONS[1:]:
        other = results[condition]["word_attr"]
        shared = [w for w in ft_attr if w in other]
        if len(shared) < 5:
            continue
        rho, p_rho = stats.spearmanr([ft_attr[w] for w in shared], [other[w] for w in shared])
        top_other = [w for w, _ in sorted(other.items(), key=lambda kv: -kv[1])[:args.top_k]]
        overlap = len(set(top_ft) & set(top_other))
        comparisons[condition] = {"spearman_rho": float(rho), "spearman_p": float(p_rho),
                                  "top_overlap": overlap, "n_shared": len(shared)}
        print(f"  {condition:20s}: Spearman rho={rho:+.3f} (p={p_rho:.3g}), "
              f"top-{args.top_k} overlap={overlap}/{args.top_k}")
    results["comparisons"] = comparisons

    print("\n[Controls] === genre-word ratio across conditions ===")
    for condition in CONDITIONS:
        s = results[condition]["genre_summary"]
        if s:
            print(f"  {condition:20s}: ratio={s['ratio']:.3f}  p={s['p']:.3g}")
    print("  (a ratio near 1.0 means genre words are not distinguished in that condition)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Controls] saved results to {args.output}")


if __name__ == "__main__":
    main()
