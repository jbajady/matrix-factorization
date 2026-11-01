import pandas as pd
from pathlib import Path

## declartion of envirment
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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




if __name__ == "__main__":
    ratings = load_ratings()
    users = load_users()
    movies = load_movies()


    print(ratings.head())
    print(users.head())
    print(movies.head())
    print(ratings.shape)
    print(users.shape)
    print(movies.shape)
