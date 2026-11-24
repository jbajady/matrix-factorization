if __package__:
    from .data_loader import load_movies, load_ratings, load_users
else:
    from data_loader import load_movies, load_ratings, load_users


def check_missing_values(ratings, users, movies):
    missing = {
        "ratings": ratings.isna().sum(),
        "users": users.isna().sum(),
        "movies": movies.isna().sum(),
    }
    for name, counts in missing.items():
        print(f"Missing values in {name}:\n{counts[counts > 0].to_string() or 'none'}")


def check_rating_range(ratings):
    minimum = ratings["Rating"].min()
    maximum = ratings["Rating"].max()
    print("Minimum rating:", minimum)
    print("Maximum rating:", maximum)
    print("Invalid ratings:", ((ratings["Rating"] < 1) | (ratings["Rating"] > 5)).sum())


def check_invalid_ids(ratings, users, movies):
    invalid_users = ~ratings["UserID"].isin(users["UserID"])
    invalid_movies = ~ratings["MovieID"].isin(movies["MovieID"])
    print("Invalid UserIDs:", int(invalid_users.sum()))
    print("Invalid MovieIDs:", int(invalid_movies.sum()))


def validate_dataset():
    ratings = load_ratings()
    users = load_users()
    movies = load_movies()
    check_missing_values(ratings, users, movies)
    check_rating_range(ratings)
    check_invalid_ids(ratings, users, movies)
    if (ratings[["UserID", "MovieID", "Rating"]].isna().any().any()
            or ((ratings["Rating"] < 1) | (ratings["Rating"] > 5)).any()
            or ratings["UserID"].isin(users["UserID"]).eq(False).any()
            or ratings["MovieID"].isin(movies["MovieID"]).eq(False).any()):
        raise ValueError("Dataset validation failed")
    return ratings, users, movies


if __name__ == "__main__":
    validate_dataset()
