# -*- coding: utf-8 -*-
"""
exCPFair Baseline Evaluator
Evaluates Recommendation Baselines (Original Paper vs Improved Tuned)
under Test-Set Reranking (evaluate.py protocol) or Full Catalog Ranking.
"""

import os
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import cornac
from cornac.eval_methods import BaseMethod
from cornac.models import MostPop, UserKNN, ItemKNN, MF, BPR, WMF, NeuMF
from cornac.metrics import NDCG, MAP, RMSE, MAE
from cornac.data import Reader

### Arguments ###
parser = argparse.ArgumentParser(description='exCPFair Baseline Evaluator (Original vs Improved)')
parser.add_argument('-d', '--dataset', help='name of dataset', required=True)
parser.add_argument('--mode', type=str, default='both', choices=['original', 'improved', 'both'], 
                    help='Evaluation mode: "original" (paper baseline), "improved" (tuned), or "both" (default: both)')
parser.add_argument('--protocol', type=str, default='reranking', choices=['reranking', 'full'],
                    help='Evaluation protocol: "reranking" (matching paper evaluate.py) or "full" (full-catalog ranking)')
parser.add_argument('-t', '--rating-threshold', type=float, default=4.0, 
                    help='Rating threshold for positive relevance in ranking metrics (default: 4.0)')
args = parser.parse_args()

# Reading train, validation, and test datasets
def read_data(dataset):
    reader = Reader()
    train_data = reader.read(fpath=f"datasets/{dataset}/train.txt", fmt='UIR', sep='\t')
    tune_data = reader.read(fpath=f"datasets/{dataset}/validation.txt", fmt='UIR', sep='\t')
    test_data = reader.read(fpath=f"datasets/{dataset}/test.txt", fmt='UIR', sep='\t')
    return train_data, tune_data, test_data

def build_original_models():
    """Exact hyperparameters from original paper script."""
    return [
        MostPop(name="MostPop"),
        UserKNN(k=20, similarity="cosine", weighting="bm25", name="UserKNN-BM25"),
        ItemKNN(k=20, similarity="cosine", name="ItemKNN-Cosine"),
        BPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=False, name="BPR"),
        WMF(k=50, max_iter=50, learning_rate=0.001, lambda_u=0.01, lambda_v=0.01, seed=123, verbose=False, name="WMF"),
        MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123, verbose=False, name="MF"),
        NeuMF(num_factors=9, layers=[32, 16, 8], act_fn="tanh", num_epochs=5, num_neg=3, batch_size=256, lr=0.001, seed=42, verbose=False, name="NeuMF")
    ]

def build_improved_models():
    """Tuned hyperparameters for superior recommendation quality."""
    return [
        MostPop(name="MostPop"),
        UserKNN(k=20, similarity="cosine", weighting="bm25", mean_centered=True, name="UserKNN-BM25"),
        ItemKNN(k=20, similarity="cosine", mean_centered=True, name="ItemKNN-Cosine"),
        BPR(k=50, max_iter=200, learning_rate=0.01, lambda_reg=0.01, verbose=False, seed=123, name="BPR"),
        WMF(k=50, max_iter=50, learning_rate=0.01, lambda_u=0.01, lambda_v=0.01, verbose=False, seed=123, name="WMF"),
        MF(k=10, max_iter=50, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123, verbose=False, name="MF"),
        NeuMF(num_factors=16, layers=[32, 16, 8], act_fn="tanh", num_epochs=20, num_neg=4, batch_size=512, lr=0.001, seed=42, verbose=False, name="NeuMF")
    ]

def evaluate_test_set_reranking(train_data, test_data, models, setting_label, rating_threshold=4.0):
    """
    Evaluates models using Test-Set Reranking (Sakai Condensed List Strategy),
    100% identical to the paper's evaluate.py metric calculation.
    """
    eval_method = BaseMethod.from_splits(
        train_data=train_data, test_data=test_data, rating_threshold=rating_threshold, exclude_unknowns=True, verbose=False
    )
    
    # Group test tuples by user ID
    user_test_records = defaultdict(list)
    for u_id, i_id, rating in test_data:
        user_test_records[u_id].append((i_id, rating))

    map_eval = MAP()
    ndcg10_eval = NDCG(k=10)
    
    results_list = []
    
    for model in tqdm(models, desc=f"Evaluating Models ({setting_label})", unit="model"):
        print(f"\n Training {model.name} [{setting_label}]...")
        model.fit(eval_method.train_set)
        
        all_true_ratings = []
        all_pred_ratings = []
        all_scaled_pred_ratings = []
        user_map_scores = []
        user_ndcg_scores = []

        is_implicit_model = isinstance(model, (BPR, WMF, NeuMF))

        for u_id, records in tqdm(user_test_records.items(), desc=f" Ranking users ({model.name})", leave=False):
            if u_id not in eval_method.train_set.uid_map:
                continue
            u_idx = eval_method.train_set.uid_map[u_id]

            valid_i_ids = []
            valid_i_idxs = []
            u_true_ratings = []

            for i_id, rating in records:
                if i_id in eval_method.train_set.iid_map:
                    valid_i_ids.append(i_id)
                    valid_i_idxs.append(eval_method.train_set.iid_map[i_id])
                    u_true_ratings.append(rating)

            if not valid_i_idxs:
                continue

            # Compute predictions
            u_pred_scores = np.array([float(np.squeeze(model.rate(u_idx, i))) for i in valid_i_idxs], dtype=np.float32)
            u_true_ratings = np.array(u_true_ratings)
            u_items = np.array(valid_i_ids)

            all_true_ratings.extend(u_true_ratings)
            all_pred_ratings.extend(u_pred_scores)

            if is_implicit_model:
                # Scale implicit probability/logits [0, 1] to 1-5 star scale for fair rating error calculation
                min_s, max_s = u_pred_scores.min(), u_pred_scores.max()
                if max_s > min_s:
                    scaled = 1.0 + 4.0 * (u_pred_scores - min_s) / (max_s - min_s)
                else:
                    scaled = np.full_like(u_pred_scores, 3.0)
                all_scaled_pred_ratings.extend(scaled)
            else:
                all_scaled_pred_ratings.extend(u_pred_scores)

            gt_pos = u_items[u_true_ratings >= rating_threshold]
            if len(gt_pos) == 0:
                continue

            u_map = map_eval.compute(item_indices=u_items, pd_scores=u_pred_scores, gt_pos=gt_pos)
            
            sorted_idx = np.argsort(-u_pred_scores)
            pd_rank = u_items[sorted_idx]
            u_ndcg = ndcg10_eval.compute(gt_pos=gt_pos, pd_rank=pd_rank)

            user_map_scores.append(u_map)
            user_ndcg_scores.append(u_ndcg)

        mae_val = float(MAE().compute(np.array(all_true_ratings), np.array(all_scaled_pred_ratings)))
        rmse_val = float(RMSE().compute(np.array(all_true_ratings), np.array(all_scaled_pred_ratings)))
        map_val = float(np.mean(user_map_scores)) if user_map_scores else 0.0
        ndcg_val = float(np.mean(user_ndcg_scores)) if user_ndcg_scores else 0.0

        results_list.append({
            "Setting": setting_label,
            "Model": model.name,
            "MAE": round(mae_val, 4),
            "RMSE": round(rmse_val, 4),
            "MAP": round(map_val, 4),
            "NDCG@10": round(ndcg_val, 4)
        })

    return results_list

def evaluate_full_catalog(train_data, test_data, models, setting_label, rating_threshold=4.0):
    """
    Evaluates models using standard Cornac full-catalog ranking.
    """
    eval_method = BaseMethod.from_splits(
        train_data=train_data, test_data=test_data, rating_threshold=rating_threshold, exclude_unknowns=True, verbose=True
    )
    metrics = [RMSE(), MAE(), MAP(), NDCG(k=10)]
    exp = cornac.Experiment(eval_method=eval_method, models=models, metrics=metrics)
    exp.run()

    results_list = []
    for r in exp.result:
        results_list.append({
            "Setting": setting_label,
            "Model": r.model_name,
            "MAE": round(r.metric_avg_results.get("MAE", 0.0), 4),
            "RMSE": round(r.metric_avg_results.get("RMSE", 0.0), 4),
            "MAP": round(r.metric_avg_results.get("MAP", 0.0), 4),
            "NDCG@10": round(r.metric_avg_results.get("NDCG@10", 0.0), 4)
        })
    return results_list

# Main Execution
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    dataset = args.dataset
    dataset_name = dataset.replace('/', '_') if '/' in dataset else dataset
    train_data, tune_data, test_data = read_data(dataset=dataset)

    all_results = []

    print("\n" + "="*80)
    print(f"EVALUATING BASELINES ON {dataset} (Protocol: {args.protocol.upper()})")
    print("="*80)

    if args.mode in ['original', 'both']:
        models_orig = build_original_models()
        if args.protocol == 'reranking':
            res_orig = evaluate_test_set_reranking(train_data, test_data, models_orig, "Original (Paper)", rating_threshold=args.rating_threshold)
        else:
            res_orig = evaluate_full_catalog(train_data, test_data, models_orig, "Original (Paper)", rating_threshold=args.rating_threshold)
        all_results.extend(res_orig)

    if args.mode in ['improved', 'both']:
        models_imp = build_improved_models()
        if args.protocol == 'reranking':
            res_imp = evaluate_test_set_reranking(train_data, test_data, models_imp, "Improved (Tuned)", rating_threshold=args.rating_threshold)
        else:
            res_imp = evaluate_full_catalog(train_data, test_data, models_imp, "Improved (Tuned)", rating_threshold=args.rating_threshold)
        all_results.extend(res_imp)

    # Save results file
    fout_path = f"results/results_{dataset_name}.txt"
    with open(fout_path, 'w', encoding='utf-8') as fout:
        header = ["Dataset", "Setting", "Model", "MAE", "RMSE", "MAP", "NDCG@10"]
        fout.write("\t".join(header) + "\n")
        for r in all_results:
            line = [dataset, r["Setting"], r["Model"], str(r["MAE"]), str(r["RMSE"]), str(r["MAP"]), str(r["NDCG@10"])]
            fout.write("\t".join(line) + "\n")

    print(f"\nSaved evaluation results to {fout_path}\n")
    print("="*85)
    print(f"EVALUATION SUMMARY ({dataset} - {args.protocol.upper()} PROTOCOL)")
    print("="*85)
    print(f"{'Model':<18} | {'Setting':<16} | {'MAE':<7} | {'RMSE':<7} | {'MAP':<7} | {'NDCG@10':<7}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['Model']:<18} | {r['Setting']:<16} | {r['MAE']:<7.4f} | {r['RMSE']:<7.4f} | {r['MAP']:<7.4f} | {r['NDCG@10']:<7.4f}")
    print("="*85)
