import numpy as np
import pandas as pd


def evaluate_rmse(
    test_ratings: pd.DataFrame,
    predictions: np.ndarray,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
) -> float:
    """Calculate RMSE for the observed user/movie pairs in a test split."""
    user_positions = {user_id: position for position, user_id in enumerate(user_ids)}
    movie_positions = {movie_id: position for position, movie_id in enumerate(movie_ids)}

    unknown_users = set(test_ratings["UserID"]) - user_positions.keys()
    unknown_movies = set(test_ratings["MovieID"]) - movie_positions.keys()
    if unknown_users or unknown_movies:
        raise ValueError("Test ratings contain users or movies absent from predictions")

    actual = test_ratings["Rating"].to_numpy(dtype=np.float64)
    estimated = np.fromiter(
        (
            predictions[user_positions[user_id], movie_positions[movie_id]]
            for user_id, movie_id in zip(
                test_ratings["UserID"], test_ratings["MovieID"]
            )
        ),
        dtype=np.float64,
        count=len(test_ratings),
    )
    return float(np.sqrt(np.mean(np.square(actual - estimated))))
