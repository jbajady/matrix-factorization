import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from utils.data_loader import load_matrix, load_ratings
from utils.data_split import split_data
from utils.evaluation import evaluate_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_svd(matrix: pd.DataFrame | np.ndarray, k: int = 50):
    values = (
        matrix.to_numpy(dtype=np.float64)
        if isinstance(matrix, pd.DataFrame)
        else np.asarray(matrix, dtype=np.float64)
    )

    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("Matrix must be two-dimensional with both dimensions >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Matrix contains NaN or infinite values")
    if not isinstance(k, int) or not 1 <= k < min(values.shape):
        raise ValueError(f"k must be an integer between 1 and {min(values.shape) - 1}")

    user_factors, singular_values, item_factors_t = svds(values, k=k)
    return user_factors, singular_values, item_factors_t


def predict_ratings(
    user_factors: np.ndarray,
    singular_values: np.ndarray,
    item_factors_t: np.ndarray,
    user_means: pd.Series,
) -> np.ndarray:
    """Reconstruct ratings and restore each user's training mean."""
    centered_predictions = (user_factors * singular_values) @ item_factors_t
    predictions = centered_predictions + user_means.to_numpy(dtype=np.float64)[:, None]
    return np.clip(predictions, 1.0, 5.0)


def run_svd(k: int = 50) -> float:
    """Train SVD from the saved matrix and evaluate on the reproducible split."""
    centered_matrix = load_matrix()
    ratings = load_ratings()
    train_ratings, test_ratings = split_data(ratings)

    user_means = (
        train_ratings.groupby("UserID")["Rating"]
        .mean()
        .reindex(centered_matrix.index)
    )
    if user_means.isna().any():
        raise ValueError("Saved matrix contains users without training ratings")

    user_ids = centered_matrix.index.to_numpy()
    movie_ids = centered_matrix.columns.to_numpy()

    user_factors, singular_values, item_factors_t = train_svd(centered_matrix, k=k)
    predictions = predict_ratings(
        user_factors, singular_values, item_factors_t, user_means
    )
    rmse = evaluate_rmse(test_ratings, predictions, user_ids, movie_ids)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPORTS_DIR / "svd_predictions.npy", predictions.astype(np.float32))

    metrics_path = REPORTS_DIR / "model_metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        if not isinstance(metrics, dict):
            raise ValueError(f"{metrics_path} must contain a JSON object")
    else:
        metrics = {}
    metrics["SVD_RMSE"] = rmse
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print(f"SVD rank: {k}")
    print(f"Training ratings: {len(train_ratings)}")
    print(f"Test ratings: {len(test_ratings)}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Predictions saved to {REPORTS_DIR / 'svd_predictions.npy'}")
    print(f"Metrics saved to {metrics_path}")
    return rmse


if __name__ == "__main__":
    run_svd()