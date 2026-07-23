# Transparent and Scrutable Recommendations Using Natural Language User Profiles

This repository is based on the official repository for the paper:

> **Transparent and Scrutable Recommendations Using Natural Language User Profiles**  
> *Jerome Ramos, Hossein A. Rahmani, Xi Wang, Xiao Fu, Aldo Lipani*  
> **ACL 2024** (Long Papers) — [ACL Anthology](https://aclanthology.org/2024.acl-long.753)

The repository has been modified to successfully reproduce the results of this paper, including updating the dependencies, making sure the same evaluation framework is used everywhere, ensuring a proper train-validation-test split and fixing minor bugs in the baseline, training and evaluation scripts. For more, please refer to the detailed reproducibility manuscript.

---

## 📌 Overview

This project introduces natural language user profiles derived from interaction histories to build **transparent, interpretable, and scrutable** recommendation systems using Large Language Models (LLMs).

---

## 📂 Datasets

The preprocessed datasets are located under `datasets/`:

* **Amazon Movies and TV**:
  ```
  datasets/Amazon/MoviesAndTV/train.jsonl
  datasets/Amazon/MoviesAndTV/validation.jsonl
  datasets/Amazon/MoviesAndTV/test.jsonl
  ```
* **TripAdvisor**:
  ```
  datasets/TripAdvisor/train.jsonl
  datasets/TripAdvisor/validation.jsonl
  datasets/TripAdvisor/test.jsonl
  ```

*For raw data source format and Sentires toolkit details, see [PETER Repository](https://github.com/lileipisces/PETER).*

---

## 👤 User Profiles

Generated natural language user profiles are located in the `user_profiles/` directory:

* `amazon_profiles.json` (Llama2-7B, 5 features)
* `amazon_profiles_mistral.json` (Mistral-7B, 5 features)
* `trip_advisor_profiles.json` (Llama2-7B, 5 features)
* `trip_advisor_profiles_mistral.json` (Mistral-7B, 5 features)

*Note: Filenames with numbers (e.g., `amazon_profiles_3.json`) represent profiles constructed using specific feature budgets.*

---

## 📊 Recommendation Baselines (`rec_baselines.py`)

Run traditional recommendation baselines (`MostPop`, `UserKNN`, `ItemKNN`, `BPR`, `WMF`, `MF`, `NeuMF`) via Cornac:

### 1. Test-Set Reranking Protocol (Matches Paper `evaluate.py`)
To evaluate baselines under the exact Test-Set Reranking protocol (Sakai Condensed List) used in the paper:

```bash
python rec_baselines.py -d Amazon/MoviesAndTV --protocol reranking
```

### 2. Compare Original Paper vs. Tuned Baselines Side-by-Side
To run both **Original Paper Hyperparameters** and **Improved Tuned Hyperparameters** in a single comparative table:

```bash
python rec_baselines.py -d Amazon/MoviesAndTV --mode both --protocol reranking
```

### 3. Full-Catalog Ranking Protocol
To evaluate baselines against the full candidate catalog (5,459 items):

```bash
python rec_baselines.py -d Amazon/MoviesAndTV --protocol full
```

---

## 🛠️ Step-by-Step Reproduction Guide

### Step 1: Preprocess Data
```bash
python preprocess.py
```

### Step 2: Generate Natural Language Profiles
Generate user profiles from interaction histories using an LLM:
```bash
python generate_profile.py
```

### Step 3: Fine-Tune Recommender LLM (`train.py`)
Fine-tune sequence classification LLMs on user profiles:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --output_dir out/amazon-out \
  --lr 0.0003 \
  --batch_size 8 \
  --num_train_epochs 5 \
  --seed 42
```

### Step 4: Evaluate Recommender Model (`evaluate.py`)
Evaluate trained LLMs on the test set:
```bash
python evaluate.py \
  --pretrained_model out/amazon-out-reproduce-profile-title \
  --profiles user_profiles/amazon_profiles.json \
  --context_in "user profile" \
  --context_out "item title"
```

### Step 5: Export LaTeX Summary Tables (`generate_latex_table.py`)
Format evaluation results into publication-ready LaTeX tables:
```bash
python generate_latex_table.py \
  --input results/evaluation_summary_comb.json \
  --output results/latex_table_comb.tex
```

---

## 🚀 Extended Experiments (Beyond Basic Reproducibility)

In addition to basic single-model reproduction, shell scripts are provided to automate batch context ablation experiments and multi-seed stability evaluations:

### 1. Batch Context Ablation Experiments (`train_comb.sh`)
Automates training and evaluation across all combinations of input contexts (`user profile`, `review history`, `item-review history`) and output formats (`item title`, `item title and description`) for Amazon Movies & TV and TripAdvisor. Upon completion, it automatically compiles the aggregated LaTeX results table:

```bash
bash train_comb.sh
```

### 2. Multi-Seed Stability & Robustness (`train_seeds.sh`)
Trains and evaluates models across 5 distinct random initialization seeds (seeds 37, 38, 39, 40, 41) to measure model variance, standard deviation, and stability across random seeds:

```bash
bash train_seeds.sh
```

---

## 🔍 Mechanistic Interpretability & Perturbation (`user_profile_interpretability.py`)

Analyze user profile semantic space, UMAP clusters, and counterfactual perturbation trajectories:

### 1. UMAP Clustering & Gradient Theme Scoring
```bash
python user_profile_interpretability.py \
  --max-profiles 100 \
  --theme-method gradient \
  --device cuda
```

### 2. Fast Profile Perturbation Trajectory
Regenerate only the profile perturbation trajectory study (`user_profile_perturbation.png`):
```bash
python user_profile_interpretability.py \
  --max-profiles 100 \
  --theme-method gradient \
  --run-perturbation \
  --device cuda
```

---

## 🧪 Scrutability & Counterfactual Test

Generate counterfactual profile edits for scrutability evaluation:
```bash
python generate_counterfactual_profiles.py
```

---

## 📜 Citation

If you find this work or code useful, please cite our ACL 2024 paper:

```bibtex
@inproceedings{ramos-etal-2024-transparent,
    title = "Transparent and Scrutable Recommendations Using Natural Language User Profiles",
    author = "Ramos, Jerome and
      Rahmani, Hossein A. and
      Wang, Xi and
      Fu, Xiao and
      Lipani, Aldo",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.753",
    pages = "13971--13984"
}
```
