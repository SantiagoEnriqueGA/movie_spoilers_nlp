import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def summary_statistics(df):
    return df.describe(include='all')

def plot_review_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=df, palette='viridis')
    plt.title('Distribution of Review Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

def plot_spoiler_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='is_spoiler', data=df, palette='viridis')
    plt.title('Distribution of Spoiler Reviews')
    plt.xlabel('Is Spoiler')
    plt.ylabel('Count')
    plt.show()

def plot_reviews_over_time(df):
    df['review_date'] = pd.to_datetime(df['review_date'])  # Ensure review_date is datetime
    df.set_index('review_date', inplace=True)  # Set review_date as index
    plt.figure(figsize=(12, 6))
    df.resample('ME').size().plot()  # Use 'ME' for month end frequency
    plt.title('Number of Reviews Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.show()

def plot_genre_distribution(df):
    genres = df['genre'].explode()
    plt.figure(figsize=(14, 7))
    sns.countplot(y=genres, order=genres.value_counts().index, palette='viridis')
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.show()

if __name__ == "__main__":
    # Load processed data
    reviews_df = load_data('data/processed/reviews.csv')
    movies_df = load_data('data/processed/movie_details.csv')

    # Generate summary statistics
    reviews_summary = summary_statistics(reviews_df)
    movies_summary = summary_statistics(movies_df)

    # Print summary statistics
    print("Reviews Summary Statistics:")
    print(reviews_summary)
    print("\nMovies Summary Statistics:")
    print(movies_summary)

    # Plot review distribution
    plot_review_distribution(reviews_df)

    # Plot spoiler distribution
    plot_spoiler_distribution(reviews_df)

    # Plot reviews over time
    plot_reviews_over_time(reviews_df)

    # Plot genre distribution
    plot_genre_distribution(movies_df)
