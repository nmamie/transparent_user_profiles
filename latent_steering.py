"""Is the genre axis present in latent space, and can we steer along it?

The perturbation trajectory in `img/user_profile_perturbation.png` edits profile *text* and
observes that the representation moves. That is two inferential steps away from the question
we care about -- whether a user's preference can be moved in latent space to change what is
recommended. This script separates the steps:

  --run-probe      Is genre linearly decodable from a profile representation? Users are
                   labelled by the dominant genre of their *actual* review history, not by
                   reading terms off a cluster, so this also validates the taste-archetype
                   labels the UMAP figure asserts.
  --run-clusters   Do the unsupervised clusters correspond to history-derived genre
                   preference (ARI / NMI), or only to lexical similarity?
  --run-steering   Causal test: inject the crime-minus-romance direction into the residual
                   stream at a given layer and measure (a) whether the probe readout moves as
                   intended and (b) whether predicted ratings and rankings follow.
  --run-checks     Verifies that the interventions propagate at all: a zero vector and a
                   constant vector must both be no-ops, a random direction must not be.
  --run-personalization
                   Does the model personalise at all? Rank each user's held-out item against
                   random catalog items.

Two implementation notes. Representations match `embed_profiles_single` in
`user_profile_interpretability.py`: mean-pooled post-`ln_f` hidden states. Interventions use
plain PyTorch forward hooks rather than nnsight, because transformers 5.x wraps block forwards
in an output-capturing decorator that silently discards nnsight write-back (reads are fine).

Additive steering of the *final* representation is provably item-independent: the head is a
single linear map, so h -> h + a*d shifts every item's prediction by a*(w.d) and cannot change
ranking. Steering therefore has to happen early enough to interact with the item tokens
through attention, which is why `--steer-layers` targets the blocks.

Example:
    python latent_steering.py --run-probe --run-clusters --run-steering --run-personalization
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from behavioral_validation import (
    GENRE_KEYWORDS,
    GENRES,
    PERTURBATION_STATES,
    build_genre_pools,
    build_prompt,
    load_item_descriptions,
    resolve_device,
)


# --------------------------------------------------------------------------- labels
def genre_of_item(title, description):
    """Exclusive keyword match, identical to the pooling rule in behavioral_validation."""
    haystack = f"{title} {description}".lower()
    hits = [g for g in GENRES if any(k in haystack for k in GENRE_KEYWORDS[g])]
    return hits[0] if len(hits) == 1 else None


def label_users_by_history(user_items, min_labelled=3, dominance=0.5):
    """Assigns each user the dominant genre of their review history.

    A user is kept only if at least `min_labelled` of their items carry a genre and one genre
    accounts for at least `dominance` of them, so labels are decisive rather than marginal.
    """
    labels = {}
    for uid, items in user_items.items():
        counts = Counter()
        for it in items:
            g = genre_of_item(it.get("title", ""), it.get("description", ""))
            if g:
                counts[g] += 1
        total = sum(counts.values())
        if total < min_labelled:
            continue
        genre, n = counts.most_common(1)[0]
        if n / total >= dominance:
            labels[uid] = genre
    return labels


# --------------------------------------------------------------- representations
def mean_pooled_representation(model, tokenizer, texts, device, batch_size=32, hook=None):
    """Mean-pooled post-ln_f hidden states, matching embed_profiles_single."""
    reps = []
    handle = None
    if hook is not None:
        module, fn = hook
        handle = module.register_forward_hook(fn)
    try:
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                hidden = model.transformer(**enc).last_hidden_state  # post ln_f
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            reps.append(pooled.float().cpu().numpy())
    finally:
        if handle is not None:
            handle.remove()
    return np.vstack(reps)


@torch.no_grad()
def predict_ratings(model, tokenizer, prompts, device, batch_size=16, max_length=300, hook=None):
    out = []
    handle = None
    if hook is not None:
        module, fn = hook
        handle = module.register_forward_hook(fn)
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.squeeze(-1)
            out.extend((logits * 4 + 1).cpu().numpy().tolist())
    finally:
        if handle is not None:
            handle.remove()
    return np.asarray(out)


def make_steering_hook(vector, span=None):
    """Adds `vector` to the residual stream, optionally only at token positions [span)."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        if span is None:
            hidden = hidden + vector
        else:
            lo, hi = span
            hidden[:, lo:hi, :] = hidden[:, lo:hi, :] + vector
        return (hidden,) + output[1:] if is_tuple else hidden
    return hook


def profile_token_span(tokenizer, prompt, profile):
    """Token index range covering the profile inside a rating prompt."""
    start = prompt.find(profile)
    end = start + len(profile)
    enc = tokenizer(prompt, return_offsets_mapping=True, truncation=True, max_length=300)
    lo, hi = None, None
    for idx, (a, b) in enumerate(enc["offset_mapping"]):
        if a == b:
            continue
        if max(a, start) < min(b, end):
            lo = idx if lo is None else lo
            hi = idx + 1
    return (lo, hi) if lo is not None else None


# ------------------------------------------------------------------- experiments
def run_checks(model, tokenizer, device):
    """Validity checks for the steering machinery, so a null result cannot be a silent no-op.

    Three properties are verified:
      1. adding a zero vector changes nothing (the hook is wired correctly);
      2. adding a *constant* vector also changes nothing, because LayerNorm removes it -- any
         steering direction must therefore be non-uniform across dimensions;
      3. additive steering after `ln_f` shifts the prediction by exactly w.d, independent of the
         item, which is why genre-selective steering has to be applied at an earlier block.
    """
    print("\n=== Steering validity checks ===")
    prompt = ('Input Context: I love crime films. Based on the input context, from a scale of 1 '
              'to 5 (1 being the lowest and 5 being the highest), I would give "Anaconda" a rating of')
    enc = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        base = float(model(**enc).logits.item())

    def logit_with(module, vec):
        handle = module.register_forward_hook(make_steering_hook(vec))
        try:
            with torch.no_grad():
                return float(model(**enc).logits.item())
        finally:
            handle.remove()

    block = model.transformer.h[6]
    zero = torch.zeros(model.config.n_embd, device=device)
    const = torch.ones(model.config.n_embd, device=device) * 0.05
    torch.manual_seed(0)
    rand = torch.randn(model.config.n_embd, device=device)
    rand = rand / rand.norm() * 5.0

    d_zero = logit_with(block, zero) - base
    d_const = logit_with(block, const) - base
    d_rand = logit_with(block, rand) - base
    print(f"  zero vector at block 6     : delta={d_zero:+.2e}  (expect 0)")
    print(f"  constant vector at block 6 : delta={d_const:+.2e}  (expect 0; removed by LayerNorm)")
    print(f"  random direction at block 6: delta={d_rand:+.2e}  (expect non-zero)")

    w = model.score.weight.detach().squeeze(0)
    observed = logit_with(model.transformer.ln_f, rand) - base
    print(f"  random direction after ln_f: delta={observed:+.6f}  "
          f"predicted w.d={float(w @ rand):+.6f}  (item-independent shift)")
    ok = abs(d_zero) < 1e-6 and abs(d_const) < 1e-4 and abs(d_rand) > 1e-6
    print(f"  -> interventions {'propagate correctly' if ok else 'FAILED validity check'}")
    return {"zero": d_zero, "constant": d_const, "random": d_rand,
            "ln_f_observed": observed, "ln_f_predicted": float(w @ rand), "passed": bool(ok)}


def run_probe(reps, labels, seed):
    """Linear probe: is genre decodable from the profile representation?"""
    print("\n=== Probe: genre decodability from profile representation ===")
    y = np.asarray(labels)
    counts = Counter(y)
    majority = max(counts.values()) / len(y)
    print(f"  n={len(y)}  class counts={dict(counts)}")
    print(f"  majority-class baseline: {majority:.3f}")

    X = StandardScaler().fit_transform(reps)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    print(f"  probe accuracy : {acc.mean():.3f} +/- {acc.std():.3f}")
    print(f"  probe macro-F1 : {f1.mean():.3f} +/- {f1.std():.3f}")

    rng = np.random.RandomState(seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    acc_shuf = cross_val_score(clf, X, y_shuf, cv=cv, scoring="accuracy")
    print(f"  shuffled-label control: {acc_shuf.mean():.3f} (sanity: should match baseline)")
    print(f"  -> lift over majority: {acc.mean() - majority:+.3f}")

    fitted = LogisticRegression(max_iter=2000).fit(X, y)
    return {
        "n": int(len(y)),
        "majority": float(majority),
        "accuracy": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "macro_f1": float(f1.mean()),
        "shuffled_accuracy": float(acc_shuf.mean()),
    }, fitted


def run_clusters(reps, labels, n_clusters, seed):
    """Do unsupervised taste clusters correspond to history-derived genre preference?"""
    print(f"\n=== Cluster validation (K={n_clusters}) vs history-derived genre ===")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(reps)
    assign = km.labels_
    y = np.asarray(labels)
    ari = adjusted_rand_score(y, assign)
    nmi = normalized_mutual_info_score(y, assign)
    print(f"  Adjusted Rand Index : {ari:.4f}   (0 = chance)")
    print(f"  Normalised MI       : {nmi:.4f}")
    print("  per-cluster genre composition (share of labelled users):")
    for c in range(n_clusters):
        m = assign == c
        if m.sum() == 0:
            continue
        comp = Counter(y[m])
        tot = m.sum()
        shares = ", ".join(f"{g}={100 * comp.get(g, 0) / tot:.0f}%" for g in GENRES)
        print(f"    cluster {c:2d} (n={tot:4d}): {shares}")
    return {"ari": float(ari), "nmi": float(nmi)}


def run_steering(model, tokenizer, args, device, title_to_desc, pools, reps, labels, probe):
    """Injects the crime-minus-romance direction and measures latent and behavioural response."""
    print("\n=== Activation steering ===")

    # analytic note, verified numerically: additive steering after ln_f is item-independent
    w = model.score.weight.detach().squeeze(0)
    print(f"  final-layer head is linear (w shape {tuple(w.shape)}): additive steering of the")
    print("  final representation shifts every item by the same a*(w.d) and cannot reorder items.")

    y = np.asarray(labels)
    base_profile = PERTURBATION_STATES["baseline"]

    results = {}
    for layer in args.steer_layers:
        block = model.transformer.h[layer]

        # direction = difference of means at this layer, over profile tokens, crime vs romance
        def layer_means(texts):
            acc, n = None, 0
            for i in range(0, len(texts), 16):
                enc = tokenizer(texts[i:i + 16], return_tensors="pt", padding=True,
                                truncation=True, max_length=300).to(device)
                captured = {}

                def cap(mod, a, out):
                    captured["h"] = out[0] if isinstance(out, tuple) else out

                h = block.register_forward_hook(cap)
                with torch.no_grad():
                    model.transformer(**enc)
                h.remove()
                hid = captured["h"]
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1)
                acc = pooled.sum(0) if acc is None else acc + pooled.sum(0)
                n += pooled.shape[0]
            return acc / n

        crime_texts = [t for t, lab in zip(args._profile_texts, y) if lab == "crime"][:args.n_direction]
        romance_texts = [t for t, lab in zip(args._profile_texts, y) if lab == "romance"][:args.n_direction]
        direction = (layer_means(crime_texts) - layer_means(romance_texts)).detach()
        print(f"\n  layer {layer}: direction from {len(crime_texts)} crime vs "
              f"{len(romance_texts)} romance profiles, ||d||={direction.norm().item():.3f}")

        layer_out = {}
        for alpha in args.alphas:
            vec = (alpha * direction).to(device)

            # (a) latent check -- does the probe readout move toward crime?
            steered_rep = mean_pooled_representation(
                model, tokenizer, [base_profile], device,
                hook=(block, make_steering_hook(vec)))
            scaler = StandardScaler().fit(reps)
            proba = probe.predict_proba(scaler.transform(steered_rep))[0]
            p_map = dict(zip(probe.classes_, proba))

            # (b) behavioural check -- ratings and ranking over the pooled candidate set
            ratings = {}
            for g in GENRES:
                prompts = [build_prompt(base_profile, t, title_to_desc) for t in pools[g]]
                span = profile_token_span(tokenizer, prompts[0], base_profile)
                ratings[g] = predict_ratings(model, tokenizer, prompts, device, args.batch_size,
                                             hook=(block, make_steering_hook(vec, span)))
            scores = np.concatenate([ratings[g] for g in GENRES])
            genre_of = np.concatenate([np.full(len(pools[g]), g) for g in GENRES])
            order = np.argsort(-scores)
            rank = np.empty(len(scores))
            rank[order] = np.arange(1, len(scores) + 1)

            mean_rank = {g: float(rank[genre_of == g].mean()) for g in GENRES}
            mean_rating = {g: float(ratings[g].mean()) for g in GENRES}
            layer_out[alpha] = {"probe": {k: float(v) for k, v in p_map.items()},
                                "mean_rating": mean_rating, "mean_rank": mean_rank}
            print(f"    a={alpha:<5} probe P(crime)={p_map.get('crime', 0):.3f} "
                  f"P(romance)={p_map.get('romance', 0):.3f} | "
                  f"r_hat crime={mean_rating['crime']:.3f} romance={mean_rating['romance']:.3f} | "
                  f"rank crime={mean_rank['crime']:.1f} romance={mean_rank['romance']:.1f}")
        results[layer] = layer_out
    return results


def run_personalization(model, tokenizer, args, device, title_to_desc):
    """Can the model rank a user's held-out item above random catalog items?"""
    print("\n=== Personalisation check (held-out item vs random items) ===")
    profiles = {p["user_id"]: p["profile"] for p in json.load(open(args.profiles, encoding="utf-8"))}
    with open(os.path.join(args.dataset_dir, "test.jsonl"), encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    all_titles = sorted(title_to_desc.keys()) if title_to_desc else None

    rng = random.Random(args.seed)
    candidates = [e for e in test if e["user"] in profiles and e.get("title")]
    rng.shuffle(candidates)
    candidates = candidates[:args.n_users_personalization]

    percentiles = []
    for e in candidates:
        profile = profiles[e["user"]]
        real = e["title"]
        distractors = rng.sample(all_titles, args.n_distractors)
        titles = [real] + [t for t in distractors if t != real]
        prompts = [build_prompt(profile, t, title_to_desc) for t in titles]
        preds = predict_ratings(model, tokenizer, prompts, device, args.batch_size)
        # percentile of the real item: fraction of distractors it outranks
        pct = float((preds[0] > preds[1:]).mean())
        percentiles.append(pct)

    percentiles = np.asarray(percentiles)
    from scipy import stats as sstats
    t, p = sstats.ttest_1samp(percentiles, 0.5)
    print(f"  users={len(percentiles)}  distractors/user={args.n_distractors}")
    print(f"  mean percentile of the real item: {percentiles.mean():.4f}  (chance = 0.5)")
    print(f"  t={t:.3f}  p={p:.4g}")
    print(f"  real item ranked #1 in {100 * (percentiles == 1.0).mean():.1f}% of cases")
    return {"mean_percentile": float(percentiles.mean()), "t": float(t), "p": float(p)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str,
                        default="out/amazon-out-reproduce-profile-title-and-description")
    parser.add_argument("--dataset_dir", type=str, default="datasets/Amazon/MoviesAndTV")
    parser.add_argument("--profiles", type=str, default="user_profiles/amazon_profiles.json")
    parser.add_argument("--user_items", type=str,
                        default="datasets/Amazon/MoviesAndTV/user_items.jsonl")
    parser.add_argument("--max_users", type=int, default=1500)
    parser.add_argument("--n_per_genre", type=int, default=110)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--n_direction", type=int, default=100)
    parser.add_argument("--steer_layers", type=int, nargs="+", default=[2, 6, 9])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--n_users_personalization", type=int, default=80)
    parser.add_argument("--n_distractors", type=int, default=99)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/latent_steering.json")
    parser.add_argument("--run-probe", dest="run_probe", action="store_true")
    parser.add_argument("--run-clusters", dest="run_clusters", action="store_true")
    parser.add_argument("--run-steering", dest="run_steering", action="store_true")
    parser.add_argument("--run-personalization", dest="run_personalization", action="store_true")
    parser.add_argument("--run-checks", dest="run_checks", action="store_true",
                        help="verify that activation interventions actually propagate")
    args = parser.parse_args()

    if not (args.run_probe or args.run_clusters or args.run_steering
            or args.run_personalization or args.run_checks):
        args.run_probe = args.run_clusters = args.run_steering = args.run_personalization = True
        args.run_checks = True

    if args.output:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Steering] model={args.model_path}  device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, problem_type="regression")
    model.eval().to(device)

    title_to_desc = load_item_descriptions(args.dataset_dir)
    pools = build_genre_pools(args.dataset_dir, args.n_per_genre, args.seed)

    user_items = json.load(open(args.user_items, encoding="utf-8"))
    profiles_list = json.load(open(args.profiles, encoding="utf-8"))
    profile_by_user = {p["user_id"]: p["profile"] for p in profiles_list}

    history_labels = label_users_by_history(user_items)
    usable = [(u, profile_by_user[u], g) for u, g in history_labels.items() if u in profile_by_user]
    random.Random(args.seed).shuffle(usable)
    usable = usable[:args.max_users]
    print(f"[Steering] users with decisive history genre and a profile: {len(usable)}")
    print("[Steering] label distribution: " + str(dict(Counter(g for _, _, g in usable))))

    profile_texts = [p for _, p, _ in usable]
    labels = [g for _, _, g in usable]
    args._profile_texts = profile_texts

    print("[Steering] extracting profile representations...")
    reps = mean_pooled_representation(model, tokenizer, profile_texts, device)
    print(f"[Steering] representations: {reps.shape}")

    results = {"model_path": args.model_path, "n_users": len(usable)}
    if args.run_checks:
        results["checks"] = run_checks(model, tokenizer, device)
    probe = None
    if args.run_probe:
        results["probe"], probe = run_probe(reps, labels, args.seed)
    if args.run_clusters:
        results["clusters"] = run_clusters(reps, labels, args.n_clusters, args.seed)
    if args.run_steering:
        if probe is None:
            _, probe = run_probe(reps, labels, args.seed)
        results["steering"] = run_steering(model, tokenizer, args, device, title_to_desc,
                                           pools, reps, labels, probe)
    if args.run_personalization:
        results["personalization"] = run_personalization(model, tokenizer, args, device, title_to_desc)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[Steering] saved results to {args.output}")


if __name__ == "__main__":
    main()
