
import pandas as pd

from data_loader import load_ratings
from data_split import split_data
from pathlib import Path

output_dir = Path("processed")
output_dir.mkdir(exist_ok=True) 


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
    # Calculate each user's average using training ratings
    user_means = matrix.mean(axis=1)

    # Center ratings around each user's average
    normalized_matrix = matrix.sub(user_means, axis=0)

    # Missing ratings become zero deviation, not zero ratings
    normalized_matrix = normalized_matrix.fillna(0)

    return normalized_matrix


if __name__ == "__main__":
    ratings = load_ratings()

    train_data, test_data = split_data(ratings)

    all_users = sorted(ratings["UserID"].unique())
    all_movies = sorted(ratings["MovieID"].unique())

    user_movie_matrix = create_user_movie_matrix(
        train_data, all_users, all_movies
    )

    normalized_matrix = normalize_user_matrix(user_movie_matrix)

    print("Original matrix shape:", user_movie_matrix.shape)
    print("Normalized matrix shape:", normalized_matrix.shape)

    print("Training sparsity:",
          user_movie_matrix.isna().sum().sum()
          / user_movie_matrix.size * 100)

    # Save compact normalized matrix
    normalized_matrix.astype("float32").round(4).to_csv(
        "processed/user_item_matrix.csv"
    )
