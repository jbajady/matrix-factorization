import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"


def _load_dat(filename: str, columns: list[str]):
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required MovieLens file not found: {path}")
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def load_ratings():
    """Load ratings and validate the core rating columns."""
    ratings = _load_dat("ratings.dat", ["UserID", "MovieID", "Rating", "Timestamp"])
    required = ["UserID", "MovieID", "Rating", "Timestamp"]
    if ratings.empty or ratings[required].isna().any().any():
        raise ValueError("ratings.dat is empty or contains missing required values")
    return ratings


def load_users():
    return _load_dat(
        "users.dat",
        ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    )


def load_movies():
    return _load_dat("movies.dat", ["MovieID", "Title", "Genres"])


def load_matrix():
    matrix_path = PROCESSED_DIR / "user_item_matrix.csv"
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Processed matrix not found: {matrix_path}. "
            "Run 'python utils/matrix_creation.py' first."
        )
    matrix = pd.read_csv(matrix_path, index_col="UserID")
    matrix.columns = pd.to_numeric(matrix.columns, errors="raise").astype("int64")
    return matrix


def load_matrix_ids():
    """Load matrix IDs without materializing all dense rating values."""
    matrix_path = PROCESSED_DIR / "user_item_matrix.csv"
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Processed matrix not found: {matrix_path}. "
            "Run 'python utils/matrix_creation.py' first."
        )
    user_ids = pd.read_csv(matrix_path, usecols=["UserID"])["UserID"].to_numpy()
    movie_ids = pd.read_csv(matrix_path, nrows=0).columns[1:].astype("int64").to_numpy()
    return user_ids, movie_ids


if __name__ == "__main__":
    ratings = load_ratings()
    users = load_users()
    movies = load_movies()
    matrix = load_matrix()

    print(ratings.head())
    print(users.head())
    print(movies.head())
    print(ratings.shape)
    print(users.shape)
    print(movies.shape)
    print(matrix.shape)
