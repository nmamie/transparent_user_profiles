# Reproduction & Extension of "Transparent and Scrutable Recommendations Using Natural Language User Profiles"

This repository provides an enhanced, out-of-the-box reproducible implementation of the paper:

> **Transparent and Scrutable Recommendations Using Natural Language User Profiles**  
> *Jerome Ramos, Hossein A. Rahmani, Xi Wang, Xiao Fu, and Aldo Lipani*  
> Published in **ACL 2024** (Long Papers)
>
> 📄 **Paper**: [ACL Anthology](https://aclanthology.org/2024.acl-long.753)  
> 🔗 **Original Codebase**: [jeromeramos70/user-profile-recommendation](https://github.com/jeromeramos70/user-profile-recommendation)

---

## 🌟 Acknowledgments & Reproducibility Enhancements

We commend Ramos et al. (2024) for their impactful contribution introducing natural language user profiles derived from interaction histories to create transparent and scrutable recommendations with Large Language Models (LLMs). Original paper source code is available at [jeromeramos70/user-profile-recommendation](https://github.com/jeromeramos70/user-profile-recommendation).

To build upon their foundation and ensure a seamless, reliable **out-of-the-box execution experience** for the research community, this repository incorporates several usability and reproducibility enhancements:

1. **Evaluation Protocol Alignment**: Standardized baseline evaluation in `rec_baselines.py` to use Test-Set Reranking (Sakai 2007 condensed list strategy), bringing baseline evaluation into exact metric parity with the paper's `evaluate.py` evaluation protocol.
2. **Automated Directory & Path Management**: Added safe, automatic creation for all output and result directories (`out/`, `results/`, `user_profiles/`), ensuring scripts run smoothly without requiring manual directory creation.
3. **Flexible & Robust CLI Interfaces**: Extended preprocessing, profile generation, training, and evaluation scripts with comprehensive command-line interfaces (`--dataset`, `--input_features`, `--output`, `--prompt_type`), eliminating the need to modify code for different datasets.
4. **Cross-Platform Compatibility**: Replaced hardcoded GPU calls with dynamic device selection (`cuda`, `cpu`, or `mps`) across training and profile generation pipelines.
5. **Comparative Hyperparameter Benchmarks**: Added `--mode {original, improved, both}` in `rec_baselines.py` to allow researchers to benchmark original paper baseline hyperparameters alongside tuned baseline configurations side-by-side.
6. **Streamlined Batch Shell Automation**: Created modular, dataset-specific batch execution scripts (`train_amazon_comb.sh`, `train_tripadvisor_comb.sh`, `train_amazon_seeds.sh`) for single-command execution of context ablation and multi-seed stability experiments.

---

## 📌 Environment Setup & Quick Start

Follow these steps to create the environment, install dependencies, and run the pipeline:

### Option A: Using `uv` (Recommended)

```bash
# 1. Initialize project structure
mkdir -p src/user_profiles_neurosymbolic_ai
touch src/user_profiles_neurosymbolic_ai/__init__.py

# 2. Create virtual environment and install dependencies with uv
uv venv .venv
uv sync

# 3. Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 4. Make shell execution scripts executable
chmod +x train_amazon_comb.sh train_tripadvisor_comb.sh train_amazon_seeds.sh

# 5. Run batch experiment scripts
./train_amazon_comb.sh
./train_tripadvisor_comb.sh
./train_amazon_seeds.sh
```

### Option B: Using Standard Python `venv` & `pip`

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Make scripts executable and run
chmod +x train_amazon_comb.sh train_tripadvisor_comb.sh train_amazon_seeds.sh
./train_amazon_comb.sh
```

---

## 📂 Datasets

Preprocessed datasets used in the paper are organized under `datasets/`:

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

*For raw data source format and Sentires toolkit details, refer to the [PETER Repository](https://github.com/lileipisces/PETER).*

---

## 👤 User Profiles

Natural language user profiles constructed from interaction histories are located in `user_profiles/`:

* `amazon_profiles.json` (Llama2-7B, 5 features)
* `amazon_profiles_mistral.json` (Mistral-7B, 5 features)
* `trip_advisor_profiles.json` (Llama2-7B, 5 features)
* `trip_advisor_profiles_mistral.json` (Mistral-7B, 5 features)

*Note: Filenames with numbers (e.g., `amazon_profiles_3.json`) represent profiles constructed using specific feature budgets.*

---

## 📊 Recommendation Baselines (`rec_baselines.py`)

Run recommendation baselines (`MostPop`, `UserKNN`, `ItemKNN`, `BPR`, `WMF`, `MF`, `NeuMF`) via Cornac:

### 1. Reproducing Paper Baseline Results (Test-Set Reranking Protocol)
To reproduce the paper's published baseline numbers under Test-Set Reranking:

```bash
python rec_baselines.py -d Amazon/MoviesAndTV --protocol reranking
```

### 2. Side-by-Side Comparison: Original Paper vs. Tuned Baselines
To evaluate both **Original Paper Hyperparameters** and **Improved Tuned Hyperparameters** side-by-side in a unified table:

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
python preprocess.py --dataset Amazon/MoviesAndTV
```

### Step 2: Generate Natural Language Profiles
Generate user profiles from interaction histories using an LLM:
```bash
python generate_profile.py --dataset Amazon/MoviesAndTV
```

### Step 3: Fine-Tune Recommender LLM (`train.py`)
Fine-tune sequence classification LLMs on user profiles:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset Amazon/MoviesAndTV \
  --profiles user_profiles/amazon_profiles.json \
  --output_dir out/amazon-out-reproduce-profile-title \
  --context_in "user profile" \
  --context_out "item title" \
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
  --dataset Amazon/MoviesAndTV \
  --profiles user_profiles/amazon_profiles.json \
  --context_in "user profile" \
  --context_out "item title" \
  --output results/profile-title.jsonl \
  --summary_file results/evaluation_summary_comb.json \
  --seed 42
```

---

## 🚀 Extended Experiments (Beyond Basic Reproducibility)

In addition to basic single-model reproduction, dedicated shell scripts automate batch context ablation experiments and multi-seed stability evaluations:

### 1. Dataset-Specific Batch Context Ablation Experiments
Automates training and evaluation across all combinations of input contexts (`user profile`, `review history`, `item-review history`) and output formats (`item title`, `item title and description`):

* **Amazon Movies & TV Dataset**:
  ```bash
  ./train_amazon_comb.sh
  ```
* **TripAdvisor Dataset**:
  ```bash
  ./train_tripadvisor_comb.sh
  ```

### 2. Multi-Seed Stability & Robustness (`train_amazon_seeds.sh`)
Trains and evaluates models across 5 distinct random initialization seeds (seeds 37, 38, 39, 40, 41) to measure model variance, standard deviation, and stability across random seeds. *Note: Multi-seed experiments were performed exclusively on the Amazon Movies & TV dataset.*

```bash
./train_amazon_seeds.sh
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

To cite the original paper by Ramos et al.:

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
