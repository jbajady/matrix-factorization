import  pandas as pd
from pathlib import Path


## check missing values in the dataframes
def check_missing_values(ratings, users, movies):
    missing_ratings = ratings.isnull().sum()
    missing_users = users.isnull().sum()
    missing_movies = movies.isnull().sum()
    print("Missing values in ratings:\n", missing_ratings)
    print("Missing values in users:\n", missing_users)
    print("Missing values in movies:\n", missing_movies)

def check_rating_range(ratings):
    print("Minimum rating:", ratings["Rating"].min())
    print("Maximum rating:", ratings["Rating"].max())

    invalid_ratings = ratings[
        (ratings["Rating"] < 1) |
        (ratings["Rating"] > 5)
    ]
    

    print("Invalid ratings:", len(invalid_ratings))

def check_invalid_ids(ratings, users, movies):
    invalid_users = ~ratings["UserID"].isin(users["UserID"])
    invalid_movies = ~ratings["MovieID"].isin(movies["MovieID"])

    print("Invalid UserIDs:", invalid_users.sum())
    print("Invalid MovieIDs:", invalid_movies.sum())

if __name__ == "__main__":
    from data_loader import load_ratings, load_users, load_movies

    # Load the data
    ratings = load_ratings()
    users = load_users()
    movies = load_movies()

    # Check for missing values
    # check_missing_values(ratings, users, movies)
    # check_rating_range(ratings)
    check_invalid_ids(ratings, users, movies)
