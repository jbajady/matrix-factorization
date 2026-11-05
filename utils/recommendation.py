from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import (
    load_matrix_ids,
    load_movies,
    load_ratings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTION_PATHS = {
    "svd": REPORTS_DIR / "svd_predictions.npy",
    "pmf": REPORTS_DIR / "pmf_predictions.npy",
}


@lru_cache(maxsize=1)
def _load_ids_and_data():
    user_ids, movie_ids = load_matrix_ids()
    ratings = load_ratings()
    movies = load_movies()
    return user_ids, movie_ids, ratings, movies


@lru_cache(maxsize=2)
def _load_predictions(model: str) -> np.ndarray:
    path = PREDICTION_PATHS[model]
    if not path.is_file():
        raise FileNotFoundError(
            f"{model.upper()} predictions not found at {path}. Train the model first."
        )
    return np.load(path, mmap_mode="r")


def generate_recommendations(user_id, model, top_n=10):
    """Return a user's highest-rated unseen movies for SVD or PMF."""
    if isinstance(user_id, bool) or not isinstance(user_id, (int, np.integer)):
        raise ValueError("user_id must be an integer")
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, (int, np.integer))
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer")
    if not isinstance(model, str) or model.lower() not in PREDICTION_PATHS:
        raise ValueError("model must be either 'svd' or 'pmf'")
    model_name = model.lower()

    user_ids, movie_ids, ratings, movies = _load_ids_and_data()
    user_positions = {value: position for position, value in enumerate(user_ids)}
    if user_id not in user_positions:
        raise ValueError(f"Unknown user ID: {user_id}")

    predictions = _load_predictions(model_name)
    if predictions.shape != (len(user_ids), len(movie_ids)):
        raise ValueError(
            f"{model_name.upper()} prediction shape {predictions.shape} does not match "
            f"the matrix IDs {(len(user_ids), len(movie_ids))}"
        )

    already_rated = set(
        ratings.loc[ratings["UserID"] == user_id, "MovieID"].to_numpy()
    )
    candidate_positions = np.fromiter(
        (
            position
            for position, movie_id in enumerate(movie_ids)
            if movie_id not in already_rated
        ),
        dtype=np.int64,
    )
    if not len(candidate_positions):
        return pd.DataFrame(
            columns=["MovieID", "Title", "Genres", "PredictedRating"]
        )

    candidate_scores = np.asarray(
        predictions[user_positions[user_id], candidate_positions]
    )
    best_positions = candidate_positions[
        np.argsort(candidate_scores)[::-1][:top_n]
    ]
    scores = np.asarray(predictions[user_positions[user_id], best_positions])
    recommendations = pd.DataFrame(
        {
            "MovieID": movie_ids[best_positions],
            "PredictedRating": scores,
        }
    ).merge(movies[["MovieID", "Title", "Genres"]], on="MovieID", how="left")
    recommendations = recommendations[
        ["MovieID", "Title", "Genres", "PredictedRating"]
    ]
    return recommendations.reset_index(drop=True)


def save_user_recommendations(user_id, top_n=10) -> Path:
    """Save both models' recommendations to the required per-user report path."""
    recommendations = []
    for model in PREDICTION_PATHS:
        model_recommendations = generate_recommendations(user_id, model, top_n)
        model_recommendations.insert(0, "Model", model.upper())
        recommendations.append(model_recommendations)
    combined_recommendations = pd.concat(recommendations, ignore_index=True)
    output_path = REPORTS_DIR / f"user_{user_id}_recommendations.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    combined_recommendations.to_csv(output_path, index=False)
    return output_path