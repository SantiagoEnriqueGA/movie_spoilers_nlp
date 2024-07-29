import pandas as pd
import os

def load_data(file_path, file_type='json', lines=True):
    """
    Loads data from a file into a pandas DataFrame.

    Args:
        file_path (str): The path to the file.
        file_type (str, optional): The type of the file. Defaults to 'json'.
        lines (bool, optional): Whether the file contains JSON objects per line. Defaults to True.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    """
    if file_type == 'json':
        return pd.read_json(file_path, lines=lines)
    return None

def save_data(df, file_path, file_type='csv'):
    """
    Saves a pandas DataFrame to a file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        file_path (str): The path to save the file.
        file_type (str, optional): The type of the file. Defaults to 'csv'.
    """
    if file_type == 'csv':
        df.to_csv(file_path, index=False)
    return None

def preprocess_reviews(reviews_df):
    """
    Preprocesses the reviews dataframe by removing duplicates, shuffling the rows,
    converting the review_date column to datetime, and encoding the user_id column as categories.
    
    Args:
        reviews_df (pandas.DataFrame): The input dataframe containing the reviews.
        
    Returns:
        pandas.DataFrame: The preprocessed dataframe.
    """
    
    reviews_df = reviews_df.drop_duplicates("review_text").sample(frac=1)       # Drop duplicates and shuffle rows
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])       # Convert review_date to datetime
    reviews_df['user_id'] = reviews_df['user_id'].astype('category').cat.codes  # Encode user_id as categories
    
    return reviews_df

def preprocess_movies(movies_df):
    """
    Preprocesses the movies dataframe by converting the 'release_date' column to datetime format.

    Args:
        movies_df (pandas.DataFrame): The input movies dataframe.

    Returns:
        pandas.DataFrame: The preprocessed movies dataframe.
    """
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='mixed') # Convert release_date to datetime
    
    return movies_df

if __name__ == "__main__":
    reviews_df = load_data('data/raw/IMDB_reviews.json')        # Load reviews data
    movies_df = load_data('data/raw/IMDB_movie_details.json')   # Load movies data

    reviews_df = preprocess_reviews(reviews_df)                 # Preprocess reviews data
    movies_df = preprocess_movies(movies_df)                    # Preprocess movies data
    
    os.makedirs('data/processed', exist_ok=True)                # Create processed data directory if it doesn't exist
    save_data(reviews_df, 'data/processed/reviews.csv')         # Save preprocessed reviews data
    save_data(movies_df, 'data/processed/movie_details.csv')    # Save preprocessed movies data
    
    # Print data shapes and heads for verification
    print("Reviews DataFrame Shape:", reviews_df.shape)
    print(reviews_df.head())
    print("Movies DataFrame Shape:", movies_df.shape)
    print(movies_df.head())
