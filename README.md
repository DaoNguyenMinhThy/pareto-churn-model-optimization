# Multi-Objective Hyperparameter Optimization for Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Optuna](https://img.shields.io/badge/Optuna-NSGA--II-green)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)

## 1. Problem Statement
In real-time banking systems, a customer churn prediction model must achieve high classification accuracy while maintaining ultra-low inference latency. Optimizing solely for predictive performance often yields overly complex ensemble models that violate production latency budgets. This project formulates hyperparameter tuning as a **Bi-Objective Optimization** problem to balance predictive power (`ROC-AUC`) and computational efficiency (`Inference Latency`).

## 2. Optimization Formulation
* **Decision Variables ($\mathbf{x}$):** LightGBM hyperparameters:
  * `n_estimators` $\in [50, 500]$ (integer)
  * `max_depth` $\in [3, 12]$ (integer)
  * `learning_rate` $\in [0.01, 0.3]$ (log-uniform)
  * `num_leaves` $\in [15, 127]$ (integer)
  * `min_child_samples` $\in [10, 100]$ (integer)
* **Objective Functions ($\mathbf{f(x)}$):**
  1. $\max f_1(\mathbf{x}) = \text{Mean 5-Fold Stratified CV ROC-AUC}$
  2. $\min f_2(\mathbf{x}) = \text{Median Inference Latency (ms per 1,000 samples)}$
* **Constraints & Evaluation Budget:**
  * Search space bounds defined above.
  * Fixed evaluation budget of $B = 40$ trials per optimization method.
  * Strict hold-out test set isolation (no test set leakage during optimization).

## 3. Dataset
* **Name:** Bank Customer Churn Prediction (`Churn_Modelling.csv`)
* **Source:** Public benchmark dataset from Kaggle (10,000 customer records, 14 attributes).
* **Split Protocol:** 80% Train-Validation (8,000 samples, evaluated via 5-Fold Stratified CV) and 20% Hold-out Test (2,000 samples) with fixed `random_state=42`.

## 4. Optimization Methodology
1. **Baseline 1 (Default Configuration):** Standard LightGBM with default library parameters.
2. **Baseline 2 (Random Search):** `optuna.samplers.RandomSampler` ($B = 40$ trials, `seed=42`).
3. **Proposed Method (NSGA-II):** Multi-objective evolutionary algorithm via `optuna.samplers.NSGAIISampler` ($B = 40$ trials, `seed=42`) to approximate the Pareto-optimal front.

## 5. Key Results
*(To be updated after completing experimental runs in Week 3-4)*

## 6. Repository Structure
```text
pareto-churn-model-optimization/
├── data/               # Dataset (Churn_Modelling.csv)
├── notebooks/          # End-to-end optimization & analysis notebooks
├── results/            # Trial logs (CSV) and summary tables
├── figures/            # Convergence plots and Pareto front visualizations
├── report/             # Final academic report (PDF) and defense slides
├── .gitignore
├── requirements.txt
└── README.md
```

## 7. How to Run
```bash
git clone [https://github.com/DaoNguyenMinhThy/pareto-churn-model-optimization.git](https://github.com/DaoNguyenMinhThy/pareto-churn-model-optimization.git)
cd pareto-churn-model-optimization
pip install -r requirements.txt
jupyter notebook notebooks/01_pareto_optimization.ipynb
```

## 8. Limitations & AI Assistance
* Inference latency measurements depend on local CPU load (mitigated using median over 50-100 repeated inference passes).
* **AI Assistance Disclosure:** Gemini was used to assist in structuring the repository documentation and experiment templates; all mathematical formulations, optimization code, and experimental validations were verified and executed by the author.