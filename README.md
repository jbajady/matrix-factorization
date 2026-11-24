# Matrix Factorization Movie Recommender

## Overview

This project compares **SVD** and **Probabilistic Matrix Factorization (PMF)** on the MovieLens 1M dataset. It provides preprocessing, reproducible training, held-out RMSE evaluation, recommendation reports, visualizations, and a Streamlit dashboard.

## Setup

Use Python 3.12 and run commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Complete workflow

Run the full pipeline once:

```bash
.venv/bin/python run_pipeline.py
```

The pipeline validates the raw data, creates the reproducible 80/20 split, writes `processed/user_item_matrix.csv`, trains SVD and PMF, evaluates both models on the same test split, and generates report artifacts.

To run stages individually:

```bash
.venv/bin/python utils/data_validation.py
.venv/bin/python utils/matrix_creation.py
.venv/bin/python -m models.svd_model
.venv/bin/python -m models.pmf_model
.venv/bin/python -m utils.model_visualization
```

The fixed split uses `random_state=42`. SVD uses rank 50. PMF uses 30 latent factors, regularized SGD, and 15 epochs.

## Dashboard

```bash
.venv/bin/python -m streamlit run app.py
```

Enter an existing MovieLens user ID to see the user's top ratings, unseen-movie recommendations from both models, and side-by-side prediction charts. Unknown IDs are reported without stopping the app.

## Report outputs

- `processed/user_item_matrix.csv`: normalized training matrix
- `reports/svd_predictions.npy`: full SVD prediction matrix
- `reports/pmf_predictions.npy`: full PMF prediction matrix
- `reports/pmf_factors/U.npy` and `V.npy`: learned latent factors
- `reports/pmf_factors/user_ids.npy` and `movie_ids.npy`: matrix IDs
- `reports/model_metrics.json`: SVD RMSE, PMF RMSE, and PMF improvement
- `reports/pmf_convergence.png`, `rmse_comparison.png`, `predicted_vs_actual.png`
- `reports/user_comparison.png`, `top_recommendations.png`
- `reports/user_<id>_recommendations.csv`: both-model recommendations

## Model behavior

SVD factorizes the centered training matrix with `scipy.sparse.linalg.svds`, reconstructs ratings, restores each user's mean, and clips predictions to 1–5.

PMF learns user and movie factors with regularized SGD over observed training ratings. The regularization term penalizes large latent factors and reduces overfitting. Its convergence plot records training MSE after each epoch.

For the current 80/20 split, the reported test RMSE values are 0.9682 for SVD and 0.8891 for PMF.
