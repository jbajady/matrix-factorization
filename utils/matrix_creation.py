
from pathlib import Path

import pandas as pd

if __package__:
    from .data_loader import load_ratings
    from .data_split import split_data
else:
    from data_loader import load_ratings
    from data_split import split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "processed" / "user_item_matrix.csv"


def create_user_movie_matrix(ratings, all_users, all_movies):
    matrix = ratings.pivot_table(
        index="UserID",
        columns="MovieID",
        values="Rating"
    )

    # Keep all users and movies from the original dataset
    matrix = matrix.reindex(
        index=all_users,
        columns=all_movies
    )

    return matrix


def normalize_user_matrix(matrix):
    normalized_matrix, _ = center_user_matrix(matrix)
    return normalized_matrix


def center_user_matrix(matrix):
    # Calculate each user's average using training ratings
    user_means = matrix.mean(axis=1)

    # Center ratings around each user's average
    normalized_matrix = matrix.sub(user_means, axis=0)

    # Missing ratings become zero deviation, not zero ratings
    normalized_matrix = normalized_matrix.fillna(0)

    return normalized_matrix, user_means


def prepare_matrix_data(ratings=None):
    """Create the shared centered train matrix and reproducible test split."""
    if ratings is None:
        ratings = load_ratings()

    required_columns = ["UserID", "MovieID", "Rating"]
    if ratings.empty or ratings[required_columns].isna().any().any():
        raise ValueError("Ratings must be non-empty and have no missing IDs or ratings")

    train_data, test_data = split_data(ratings)

    all_users = sorted(ratings["UserID"].unique())
    all_movies = sorted(ratings["MovieID"].unique())

    user_movie_matrix = create_user_movie_matrix(
        train_data, all_users, all_movies
    )

    normalized_matrix, user_means = center_user_matrix(user_movie_matrix)
    return normalized_matrix, user_means, train_data, test_data


def save_user_item_matrix(matrix):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.astype("float32").round(4).to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    normalized_matrix, _, train_data, _ = prepare_matrix_data()

    print("Normalized matrix shape:", normalized_matrix.shape)
    print(f"Training ratings: {len(train_data)}")

    # Save compact normalized matrix
    save_user_item_matrix(normalized_matrix)
    print(f"Saved matrix to {OUTPUT_PATH}")
