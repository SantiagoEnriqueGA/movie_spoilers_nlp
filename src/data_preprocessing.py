import pandas as pd
import os

def load_data(file_path, file_type='json', lines=True):
    if file_type == 'json':
        return pd.read_json(file_path, lines=lines)
    return None

def preprocess_reviews(reviews_df):
    # Remove duplicates and shuffle
    reviews_df = reviews_df.drop_duplicates("review_text").sample(frac=1)
    
    # Convert review_date to datetime
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    
    # Encode user_id as category
    reviews_df['user_id'] = reviews_df['user_id'].astype('category').cat.codes
    
    return reviews_df

def preprocess_movies(movies_df):
    # Convert release_date to datetime
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format='mixed')
    
    return movies_df

def save_data(df, file_path, file_type='csv'):
    if file_type == 'csv':
        df.to_csv(file_path, index=False)
    return None

if __name__ == "__main__":
    # Load data
    reviews_df = load_data('data/raw/IMDB_reviews.json')
    movies_df = load_data('data/raw/IMDB_movie_details.json')

    # Preprocess data
    reviews_df = preprocess_reviews(reviews_df)
    movies_df = preprocess_movies(movies_df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    save_data(reviews_df, 'data/processed/reviews.csv')
    save_data(movies_df, 'data/processed/movie_details.csv')
    
    # Print data shapes and heads for verification
    print("Reviews DataFrame Shape:", reviews_df.shape)
    print(reviews_df.head())
    print("Movies DataFrame Shape:", movies_df.shape)
    print(movies_df.head())
