
import pandas as pd
from pathlib import Path

## declartion of envirment
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR_CSV = Path(__file__).resolve().parent.parent / "processed"


## functions to load data 


def load_ratings():
    return pd.read_csv(DATA_DIR / "ratings.dat", sep="::",
    engine='python',
     names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
     encoding='latin-1')


def load_users():
    return pd.read_csv(DATA_DIR / "users.dat", sep='::',
     engine='python', 
     names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
     encoding='latin-1')


def load_movies():
    return pd.read_csv(DATA_DIR / "movies.dat", sep='::',
     engine='python',
     names=['MovieID', 'Title', 'Genres'],
     encoding='latin-1')


def load_matrix():
    matrix = pd.read_csv(DATA_DIR_CSV / "user_item_matrix.csv", index_col="UserID")
    matrix.columns = pd.to_numeric(matrix.columns, errors="raise").astype("int64")
    return matrix


def load_matrix_ids():
    """Load matrix user and movie IDs without parsing the dense rating values."""
    matrix_path = DATA_DIR_CSV / "user_item_matrix.csv"
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
