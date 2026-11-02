from sklearn.model_selection import train_test_split
from data_loader import load_ratings


def split_data(ratings):
    train_data, test_data = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42
    )

    return train_data, test_data


if __name__ == "__main__":
    ratings = load_ratings()

    train_data, test_data = split_data(ratings)

    print("Training size:", len(train_data))
    print("Testing size:", len(test_data))