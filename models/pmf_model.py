import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_loader import load_matrix, load_ratings
from utils.data_split import split_data
from utils.evaluation import evaluate_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FACTORS_DIR = REPORTS_DIR / "pmf_factors"


def _training_mse(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_positions: np.ndarray,
    item_positions: np.ndarray,
    centered_ratings: np.ndarray,
) -> float:
    squared_error = 0.0
    for start in range(0, len(centered_ratings), 65_536):
        end = start + 65_536
        estimates = np.einsum(
            "ij,ij->i",
            user_factors[user_positions[start:end]],
            item_factors[item_positions[start:end]],
        )
        errors = centered_ratings[start:end] - estimates
        squared_error += float(np.dot(errors, errors))
    return squared_error / len(centered_ratings)


def train_pmf(
    train_ratings,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    n_factors: int = 30,
    epochs: int = 15,
    learning_rate: float = 0.03,
    regularization: float = 0.02,
    batch_size: int = 4096,
    random_state: int = 42,
):
    """Train probabilistic matrix factorization with regularized mini-batch SGD."""
    if train_ratings.empty:
        raise ValueError("Training ratings are empty")
    if not isinstance(n_factors, int) or n_factors < 1:
        raise ValueError("n_factors must be a positive integer")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0 or regularization < 0 or batch_size < 1:
        raise ValueError("Learning rate and batch size must be positive; regularization non-negative")

    user_index = {user_id: position for position, user_id in enumerate(user_ids)}
    movie_index = {movie_id: position for position, movie_id in enumerate(movie_ids)}
    try:
        users = train_ratings["UserID"].map(user_index).to_numpy(dtype=np.int64)
        items = train_ratings["MovieID"].map(movie_index).to_numpy(dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("Training ratings contain IDs absent from the model matrix") from error
    if train_ratings[["UserID", "MovieID", "Rating"]].isna().any().any():
        raise ValueError("Training ratings contain missing values")
    if (train_ratings["UserID"].map(user_index).isna().any()
            or train_ratings["MovieID"].map(movie_index).isna().any()):
        raise ValueError("Training ratings contain IDs absent from the model matrix")

    ratings = train_ratings["Rating"].to_numpy(dtype=np.float64)
    global_mean = float(ratings.mean())
    centered_ratings = ratings - global_mean
    rng = np.random.default_rng(random_state)
    user_factors = rng.normal(0.0, 0.1, size=(len(user_ids), n_factors))
    item_factors = rng.normal(0.0, 0.1, size=(len(movie_ids), n_factors))
    mse_history = []

    for epoch in range(epochs):
        permutation = rng.permutation(len(ratings))
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            batch_users = users[batch]
            batch_items = items[batch]
            batch_targets = centered_ratings[batch]

            old_user_factors = user_factors[batch_users].copy()
            old_item_factors = item_factors[batch_items].copy()
            errors = batch_targets - np.einsum(
                "ij,ij->i", old_user_factors, old_item_factors
            )
            user_gradients = errors[:, None] * old_item_factors
            item_gradients = errors[:, None] * old_user_factors

            unique_users, user_inverse, user_counts = np.unique(
                batch_users, return_inverse=True, return_counts=True
            )
            unique_items, item_inverse, item_counts = np.unique(
                batch_items, return_inverse=True, return_counts=True
            )
            user_gradient_sums = np.zeros((len(unique_users), n_factors))
            item_gradient_sums = np.zeros((len(unique_items), n_factors))
            np.add.at(user_gradient_sums, user_inverse, user_gradients)
            np.add.at(item_gradient_sums, item_inverse, item_gradients)

            user_factors[unique_users] += learning_rate * (
                user_gradient_sums / user_counts[:, None]
                - regularization * user_factors[unique_users]
            )
            item_factors[unique_items] += learning_rate * (
                item_gradient_sums / item_counts[:, None]
                - regularization * item_factors[unique_items]
            )

        mse = _training_mse(
            user_factors, item_factors, users, items, centered_ratings
        )
        mse_history.append(mse)
        print(f"Epoch {epoch + 1}/{epochs} - training MSE: {mse:.6f}")

    return user_factors, item_factors, global_mean, mse_history


def run_pmf(
    n_factors: int = 30,
    epochs: int = 15,
    learning_rate: float = 0.03,
    regularization: float = 0.02,
    batch_size: int = 4096,
) -> float:
    """Train PMF, evaluate on the held-out split, and save model artifacts."""
    matrix = load_matrix()
    ratings = load_ratings()
    train_ratings, test_ratings = split_data(ratings)
    user_ids = matrix.index.to_numpy()
    movie_ids = matrix.columns.to_numpy()

    user_factors, item_factors, global_mean, mse_history = train_pmf(
        train_ratings,
        user_ids,
        movie_ids,
        n_factors=n_factors,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        batch_size=batch_size,
    )
    predictions = np.clip(
        global_mean + user_factors @ item_factors.T, 1.0, 5.0
    )
    rmse = evaluate_rmse(test_ratings, predictions, user_ids, movie_ids)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPORTS_DIR / "pmf_predictions.npy", predictions.astype(np.float32))
    np.save(FACTORS_DIR / "U.npy", user_factors.astype(np.float32))
    np.save(FACTORS_DIR / "V.npy", item_factors.astype(np.float32))
    np.save(FACTORS_DIR / "user_ids.npy", user_ids)
    np.save(FACTORS_DIR / "movie_ids.npy", movie_ids)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(1, epochs + 1), mse_history, marker="o")
    axis.set(title="PMF Training Convergence", xlabel="Epoch", ylabel="Training MSE")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "pmf_convergence.png", dpi=150)
    plt.close(figure)

    metrics_path = REPORTS_DIR / "model_metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        if not isinstance(metrics, dict):
            raise ValueError(f"{metrics_path} must contain a JSON object")
    else:
        metrics = {}
    metrics["PMF_RMSE"] = rmse
    if isinstance(metrics.get("SVD_RMSE"), (int, float)):
        metrics["PMF_vs_SVD_improvement_%"] = (
            (metrics["SVD_RMSE"] - rmse) / metrics["SVD_RMSE"] * 100
        )
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print(f"PMF test RMSE: {rmse:.4f}")
    print(f"PMF predictions saved to {REPORTS_DIR / 'pmf_predictions.npy'}")
    print(f"PMF factors saved under {FACTORS_DIR}")
    print(f"Convergence plot saved to {REPORTS_DIR / 'pmf_convergence.png'}")
    return rmse


if __name__ == "__main__":
    run_pmf()