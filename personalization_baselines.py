"""Does anything personalise on this data, or is the task itself the problem?

`latent_steering.py --run-personalization` found that the fine-tuned profile model cannot rank a
user's held-out item above random catalog items. That result is only interpretable against a
control: if classical collaborative-filtering baselines also fail on the same candidate sets,
the limitation lies in the data or the protocol rather than in the profile model.

This script scores identical candidate sets -- one held-out item per user plus `--n_distractors`
items the user has not interacted with -- with cornac baselines and with the fine-tuned profile
model, and reports the percentile at which each ranker places the real item (0.5 = chance).

Note on the evaluation protocol this contrasts with: the paper's reported NDCG/MAP use test-set
reranking (Sakai condensed list), which orders only items the user already chose. That is a
strictly easier task than the one measured here, and is achievable from item-side priors alone.

Example:
    python personalization_baselines.py --dataset Amazon/MoviesAndTV --n_users 200
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from scipy import stats

import cornac
from cornac.data import Reader
from cornac.eval_methods import BaseMethod
from cornac.models import MF, BPR, WMF, MostPop
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from behavioral_validation import build_prompt, load_item_descriptions, resolve_device


def build_models(mode):
    """Hyperparameters lifted from rec_baselines.py so the control matches the paper's setup."""
    if mode == "original":
        return [
            MostPop(name="MostPop"),
            BPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=False, name="BPR"),
            WMF(k=50, max_iter=50, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01, seed=123,
                verbose=False, name="WMF"),
            MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123,
               verbose=False, name="MF"),
        ]
    return [
        MostPop(name="MostPop"),
        BPR(k=50, max_iter=200, learning_rate=0.01, lambda_reg=0.01, verbose=False, seed=123, name="BPR"),
        WMF(k=50, max_iter=50, learning_rate=0.01, lambda_u=0.01, lambda_v=0.01, verbose=False,
            seed=123, name="WMF"),
        MF(k=10, max_iter=50, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123,
           verbose=False, name="MF"),
    ]


@torch.no_grad()
def llm_scores(model, tokenizer, profile, titles, title_to_desc, device, batch_size=25):
    prompts = [build_prompt(profile, t, title_to_desc) for t in titles]
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], truncation=True, padding="max_length",
                        max_length=300, return_tensors="pt").to(device)
        out.extend(model(**enc).logits.squeeze(-1).cpu().numpy().tolist())
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="Amazon/MoviesAndTV")
    parser.add_argument("--model_path", type=str,
                        default="out/amazon-out-reproduce-profile-title-and-description")
    parser.add_argument("--profiles", type=str, default="user_profiles/amazon_profiles.json")
    parser.add_argument("--mode", type=str, default="improved", choices=["original", "improved"])
    parser.add_argument("--n_users", type=int, default=200)
    parser.add_argument("--n_distractors", type=int, default=99)
    parser.add_argument("--rating_threshold", type=float, default=4.0)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/personalization_baselines.json")
    args = parser.parse_args()

    if args.output:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    data_dir = f"datasets/{args.dataset}"
    reader = Reader()
    train_data = reader.read(fpath=f"{data_dir}/train.txt", fmt="UIR", sep="\t")
    test_data = reader.read(fpath=f"{data_dir}/test.txt", fmt="UIR", sep="\t")

    eval_method = BaseMethod.from_splits(
        train_data=train_data, test_data=test_data,
        rating_threshold=args.rating_threshold, exclude_unknowns=True, verbose=False)
    train_set = eval_method.train_set
    uid_map, iid_map = train_set.uid_map, train_set.iid_map
    idx_to_raw_item = {v: k for k, v in iid_map.items()}
    print(f"[Baselines] train users={train_set.num_users} items={train_set.num_items}")

    # items each user already interacted with in train, to exclude from distractors
    seen = {u: set(train_set.csr_matrix.getrow(u).indices) for u in range(train_set.num_users)}

    id_to_title = {}
    with open(f"{data_dir}/item.json", encoding="utf-8") as f:
        for it in json.load(f):
            if it.get("item") and it.get("title"):
                id_to_title.setdefault(it["item"], it["title"])

    # candidate sets: one held-out positive per user + distractors, built once and shared
    test_by_user = {}
    for raw_u, raw_i, rating in test_data:
        if raw_u in uid_map and raw_i in iid_map and rating >= args.rating_threshold:
            test_by_user.setdefault(raw_u, []).append(raw_i)

    profiles = {p["user_id"]: p["profile"] for p in json.load(open(args.profiles, encoding="utf-8"))}
    eligible = [u for u in test_by_user if u in profiles and
                all(idx_to_raw_item[i] in id_to_title for i in [iid_map[test_by_user[u][0]]])]
    rng.shuffle(eligible)
    eligible = eligible[:args.n_users]
    print(f"[Baselines] users evaluated: {len(eligible)}")

    all_item_idx = list(range(train_set.num_items))
    cases = []
    for raw_u in eligible:
        u = uid_map[raw_u]
        pos_raw = rng.choice(test_by_user[raw_u])
        pos = iid_map[pos_raw]
        pool = [i for i in rng.sample(all_item_idx, args.n_distractors * 3)
                if i != pos and i not in seen[u] and idx_to_raw_item[i] in id_to_title]
        distractors = pool[:args.n_distractors]
        if len(distractors) < args.n_distractors:
            continue
        cases.append({"raw_u": raw_u, "u": u, "pos": pos, "cands": [pos] + distractors})
    print(f"[Baselines] usable cases: {len(cases)}")

    results = {}

    for model in build_models(args.mode):
        print(f"\n[Baselines] fitting {model.name} ...")
        model.fit(train_set)
        pcts = []
        for case in cases:
            try:
                scores = model.score(case["u"])
                s = np.asarray([scores[i] for i in case["cands"]], dtype=float)
            except Exception:
                s = np.asarray([float(model.score(case["u"], i)) for i in case["cands"]])
            pcts.append(float((s[0] > s[1:]).mean()))
        pcts = np.asarray(pcts)
        t, p = stats.ttest_1samp(pcts, 0.5)
        results[model.name] = {"mean_percentile": float(pcts.mean()), "t": float(t), "p": float(p),
                               "top1_rate": float((pcts == 1.0).mean())}
        print(f"  {model.name:10s} percentile={pcts.mean():.4f}  t={t:.2f}  p={p:.3g}  "
              f"top-1={100 * (pcts == 1.0).mean():.1f}%")

    if not args.skip_llm:
        device = resolve_device(args.device)
        print(f"\n[Baselines] scoring the fine-tuned profile model on the same sets ({device}) ...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        llm = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, problem_type="regression").eval().to(device)
        title_to_desc = load_item_descriptions(data_dir)

        pcts = []
        for n, case in enumerate(cases, 1):
            titles = [id_to_title[idx_to_raw_item[i]] for i in case["cands"]]
            s = llm_scores(llm, tokenizer, profiles[case["raw_u"]], titles, title_to_desc, device)
            pcts.append(float((s[0] > s[1:]).mean()))
            if n % 50 == 0:
                print(f"    {n}/{len(cases)} users scored")
        # Diagnostic: MostPop scores well above chance here, so the candidate sets carry a
        # popularity signal. Does the profile model capture any of it?
        popularity = np.zeros(train_set.num_items)
        for u in range(train_set.num_users):
            for i in train_set.csr_matrix.getrow(u).indices:
                popularity[i] += 1
        probe_case = cases[0]
        probe_items = rng.sample(range(train_set.num_items), 300)
        probe_titles = [id_to_title[idx_to_raw_item[i]] for i in probe_items]
        probe_scores = llm_scores(llm, tokenizer, profiles[probe_case["raw_u"]],
                                  probe_titles, title_to_desc, device)
        rho, p_rho = stats.spearmanr(probe_scores, [popularity[i] for i in probe_items])
        pos_pop = float(np.mean([popularity[c["pos"]] for c in cases]))
        dis_pop = float(np.mean([popularity[i] for c in cases for i in c["cands"][1:]]))
        print(f"  diagnostic: train interactions for positives={pos_pop:.1f} "
              f"vs distractors={dis_pop:.1f}")
        print(f"  diagnostic: Spearman(profile-model score, item popularity) = {rho:+.3f} "
              f"p={p_rho:.3g}")

        pcts = np.asarray(pcts)
        t, p = stats.ttest_1samp(pcts, 0.5)
        results["popularity_diagnostic"] = {"spearman_rho": float(rho), "p": float(p_rho),
                                            "positive_popularity": pos_pop,
                                            "distractor_popularity": dis_pop}
        results["ProfileLLM"] = {"mean_percentile": float(pcts.mean()), "t": float(t), "p": float(p),
                                 "top1_rate": float((pcts == 1.0).mean())}
        print(f"  {'ProfileLLM':10s} percentile={pcts.mean():.4f}  t={t:.2f}  p={p:.3g}  "
              f"top-1={100 * (pcts == 1.0).mean():.1f}%")

    print("\n=== percentile of the held-out item (0.5 = chance) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_percentile"]):
        flag = "" if r["p"] < 0.05 else "   (n.s.)"
        print(f"  {name:12s} {r['mean_percentile']:.4f}  p={r['p']:.3g}{flag}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"n_cases": len(cases), "mode": args.mode, "results": results}, f, indent=2)
        print(f"\n[Baselines] saved results to {args.output}")


if __name__ == "__main__":
    main()
