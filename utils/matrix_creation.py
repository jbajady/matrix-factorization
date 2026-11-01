from data_loader import load_ratings, load_users, load_movies
 

## function to create user-movie matrix 
def create_user_movie_matrix(ratings, users, movies):
    """
    Create a user-movie matrix from the ratings, users, and movies dataframes.

    Parameters:
    ratings (pd.DataFrame): DataFrame containing user ratings.
    users (pd.DataFrame): DataFrame containing user information.
    movies (pd.DataFrame): DataFrame containing movie information.

    Returns:
    pd.DataFrame: A user-movie matrix with users as rows and movies as columns.
    """
    # Merge ratings with users and movies to get complete information
    merged_data = ratings.merge(users, on='UserID').merge(movies, on='MovieID')

    # Create the user-movie matrix using pivot_table
    user_movie_matrix = merged_data.pivot_table(index='UserID', columns='MovieID', values='Rating')

    return user_movie_matrix


if __name__ == "__main__":

    # Load the data
    ratings = load_ratings()
    users = load_users()
    movies = load_movies()

    # Create the user-movie matrix
    user_movie_matrix = create_user_movie_matrix(ratings, users, movies)
    total_cells = user_movie_matrix.size

    missing_cells = user_movie_matrix.isna().sum().sum()
    
    sparsity = (missing_cells / total_cells) * 100
    
    print("Sparsity:", sparsity, "%")

    # Display the user-movie matrix
    print(user_movie_matrix.shape)