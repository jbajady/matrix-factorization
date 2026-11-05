import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_loader import load_matrix_ids, load_ratings
from utils.data_split import split_data
from utils.evaluation import evaluate_rmse
from utils.recommendation import (
    PREDICTION_PATHS,
    generate_recommendations,
    save_user_recommendations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_NAMES = ("svd", "pmf")


def _prediction_values(
    ratings: pd.DataFrame,
    predictions: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> np.ndarray:
    user_positions = pd.Series(np.arange(len(user_ids)), index=user_ids)
    movie_positions = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    users = ratings["UserID"].map(user_positions)
    movies = ratings["MovieID"].map(movie_positions)
    if users.isna().any() or movies.isna().any():
        raise ValueError("Ratings contain IDs absent from the prediction matrix")
    return np.asarray(
        predictions[users.to_numpy(dtype=np.int64), movies.to_numpy(dtype=np.int64)]
    )


def _save_prediction_comparison(
    test_ratings: pd.DataFrame,
    predictions_by_model: dict[str, np.ndarray],
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    max_points: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    actual = test_ratings["Rating"].to_numpy(dtype=np.float64)
    sample_size = min(max_points, len(test_ratings))
    sample_positions = np.random.default_rng(42).choice(
        len(test_ratings), size=sample_size, replace=False
    )

    for axis, model in zip(axes, MODEL_NAMES):
        estimates = _prediction_values(
            test_ratings, predictions_by_model[model], user_ids, movie_ids
        )
        axis.scatter(
            actual[sample_positions],
            estimates[sample_positions],
            alpha=0.2,
            s=8,
            rasterized=True,
        )
        axis.plot([1, 5], [1, 5], color="black", linestyle="--", linewidth=1)
        axis.set(
            title=f"{model.upper()} (RMSE {evaluate_rmse(test_ratings, predictions_by_model[model], user_ids, movie_ids):.3f})",
            xlabel="Actual rating",
            ylabel="Predicted rating",
            xlim=(1, 5),
            ylim=(1, 5),
        )
        axis.grid(True, alpha=0.2)

    figure.suptitle("Predicted vs. Actual Ratings (held-out test set)")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(figure)


def _save_rmse_comparison(metrics: dict[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    models = ("SVD", "PMF")
    values = [metrics[f"{model}_RMSE"] for model in models]
    bars = axis.bar(models, values, color=["#4C78A8", "#F58518"])
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set(title="Test RMSE Comparison", ylabel="RMSE (lower is better)")
    axis.set_ylim(0, max(values) * 1.2)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "rmse_comparison.png", dpi=150)
    plt.close(figure)


def _save_user_comparison(user_id: int, top_n: int) -> None:
    recommendations = {
        model: generate_recommendations(user_id, model, top_n)
        for model in MODEL_NAMES
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(16, max(5, top_n * 0.42)), sharex=True
    )
    for axis, model in zip(axes, MODEL_NAMES):
        ranked = recommendations[model].sort_values("PredictedRating")
        axis.barh(ranked["Title"], ranked["PredictedRating"],
                  color="#4C78A8" if model == "svd" else "#F58518")
        axis.set(
            title=f"{model.upper()} top {len(ranked)} recommendations",
            xlabel="Predicted rating",
            xlim=(1, 5),
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(f"Recommendation Comparison for User {user_id}")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "user_comparison.png", dpi=150)
    plt.close(figure)

    all_movies = sorted(
        set(recommendations["svd"]["MovieID"])
        | set(recommendations["pmf"]["MovieID"])
    )
    titles = (
        pd.concat(recommendations.values())
        .drop_duplicates("MovieID")
        .set_index("MovieID")["Title"]
        .reindex(all_movies)
    )
    positions = np.arange(len(all_movies))
    bar_height = 0.38
    figure, axis = plt.subplots(figsize=(12, max(5, len(all_movies) * 0.35)))
    for offset, model, color in (
        (-bar_height / 2, "svd", "#4C78A8"),
        (bar_height / 2, "pmf", "#F58518"),
    ):
        scores = recommendations[model].set_index("MovieID")["PredictedRating"]
        values = scores.reindex(all_movies).to_numpy(dtype=np.float64)
        axis.barh(positions + offset, values, height=bar_height, label=model.upper(),
                  color=color)
    axis.set_yticks(positions, titles.to_list())
    axis.set(
        title=f"Top Recommendations by Model for User {user_id}",
        xlabel="Predicted rating",
        xlim=(1, 5),
    )
    axis.invert_yaxis()
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "top_recommendations.png", dpi=150)
    plt.close(figure)


def generate_model_visualizations(
    user_id: int = 1,
    top_n: int = 10,
    max_points: int = 20_000,
) -> dict[str, float]:
    """Create model evaluation plots and a selected user's recommendation plots."""
    if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
        raise ValueError("user_id must be an integer")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    user_ids, movie_ids = load_matrix_ids()
    train_ratings, test_ratings = split_data(load_ratings())
    predictions_by_model = {}
    metrics = {}
    for model in MODEL_NAMES:
        prediction_path = PREDICTION_PATHS[model]
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"{model.upper()} predictions not found at {prediction_path}. "
                "Train both models before generating comparisons."
            )
        predictions = np.load(prediction_path, mmap_mode="r")
        if predictions.shape != (len(user_ids), len(movie_ids)):
            raise ValueError(
                f"{model.upper()} prediction shape {predictions.shape} does not match "
                f"the saved matrix IDs {(len(user_ids), len(movie_ids))}"
            )
        predictions_by_model[model] = predictions
        metrics[f"{model.upper()}_RMSE"] = evaluate_rmse(
            test_ratings, predictions, user_ids, movie_ids
        )

    metrics["PMF_vs_SVD_improvement_%"] = (
        (metrics["SVD_RMSE"] - metrics["PMF_RMSE"])
        / metrics["SVD_RMSE"]
        * 100
    )
    _save_prediction_comparison(
        test_ratings, predictions_by_model, user_ids, movie_ids, max_points
    )
    _save_rmse_comparison(metrics)
    _save_user_comparison(user_id, top_n)
    save_user_recommendations(user_id, top_n)

    metrics_path = REPORTS_DIR / "model_metrics.json"
    if metrics_path.exists() and metrics_path.stat().st_size:
        with metrics_path.open(encoding="utf-8") as metrics_file:
            existing_metrics = json.load(metrics_file)
        if not isinstance(existing_metrics, dict):
            raise ValueError(f"{metrics_path} must contain a JSON object")
        existing_metrics.update(metrics)
        metrics = existing_metrics
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print(f"Test SVD RMSE: {metrics['SVD_RMSE']:.4f}")
    print(f"Test PMF RMSE: {metrics['PMF_RMSE']:.4f}")
    print(f"Comparison plots saved under {REPORTS_DIR}")
    print(f"Recommendations saved for user {user_id}")
    return metrics


if __name__ == "__main__":
    generate_model_visualizations()
