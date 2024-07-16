import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
import multiprocessing as mp
from datetime import datetime

# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Function for multiprocessing
def process_text_features(args):
    review_text, sid, keywords = args
    word_count = len(word_tokenize(review_text))
    avg_word_len = avg_word_length(review_text)
    num_sentences = len(sent_tokenize(review_text))
    compound_sentiment = sid.polarity_scores(review_text)['compound']
    keyword_flags = [int(keyword in review_text.lower()) for keyword in keywords]
    return word_count, avg_word_len, num_sentences, compound_sentiment, keyword_flags

def avg_word_length(text):
    words = word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

# Function to convert duration to minutes
def convert_duration_to_minutes(duration):
    match = re.match(r'(\d+)h (\d+)min', duration)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
    else:
        return 0

if __name__ == "__main__":
    # Load processed data
    reviews_df = pd.read_csv('data/processed/reviews.csv')
    movies_df = pd.read_csv('data/processed/movie_details.csv')
    reviews_df.shape
    movies_df.shape

    # -----------------------------
    import time
    start_time_single = time.time()
    # -----------------------------

    print('Text-Based Features')
    # 1. Text-Based Features 
    # ------------------------------------------------------------------------------------------------------------


    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Keywords for feature extraction
    keywords = ['romance', 'action', 'comedy', 'thriller', 'drama', 'horror', 'sci-fi', 'fantasy', 'romantic', 'violence']

    # Prepare arguments for multiprocessing
    args = [(review_text, sid, keywords) for review_text in reviews_df['review_text']]

    # Parallel processing for other text features
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_text_features, args)

    # Unpack the results into the DataFrame
    reviews_df['word_count'], reviews_df['avg_word_length'], reviews_df['num_sentences'], reviews_df['compound_sentiment_score'], keyword_flags = zip(*results)

    # Add keyword flags to DataFrame
    for i, keyword in enumerate(keywords):
        reviews_df[f'has_{keyword}'] = [flags[i] for flags in keyword_flags]


    print('Date-Based Features')
    # 2. Date-Based Features
    # ------------------------------------------------------------------------------------------------------------

    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    reviews_df['review_year'] = reviews_df['review_date'].dt.year
    reviews_df['review_month'] = reviews_df['review_date'].dt.month
    reviews_df['review_day'] = reviews_df['review_date'].dt.day

    print('Movie Details Features')
    # 3. Movie Details Features
    # ------------------------------------------------------------------------------------------------------------

    # Duration feature (assuming it's available in movie_details.csv)
    movies_df['duration_minutes'] = movies_df['duration'].apply(convert_duration_to_minutes)

    # Genre encoding (assuming 'genre' is categorical)
    # movies_df = pd.get_dummies(movies_df, columns=['genre'], prefix='genre')

    # Merge reviews_df and movies_df if necessary based on 'movie_id'
    merged_df = pd.merge(reviews_df, movies_df, on='movie_id', how='inner')


    print('Statistical Features')
    # 4. Statistical Features
    # ------------------------------------------------------------------------------------------------------------

    rating_stats = reviews_df.groupby('movie_id')['rating'].agg(['mean', 'median', 'std']).reset_index()
    rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']

    # Merge rating_stats with merged_df on 'movie_id'
    final_df = pd.merge(merged_df, rating_stats, on='movie_id', how='left')
    
    
    print('TF-IDF Features')
    # 5. TF-IDF Features
    # ------------------------------------------------------------------------------------------------------------
    # TF-IDF Features
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(final_df['review_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    # final_df = pd.concat([final_df, tfidf_df], axis=1)

    # SVD for dimensionality reduction of TF-IDF features
    svd = TruncatedSVD(n_components=100)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    svd_df = pd.DataFrame(svd_matrix, columns=[f'svd_{i}' for i in range(100)])
    final_df = pd.concat([final_df, svd_df], axis=1)
    

    print('Saving')
    # Save the engineered datasets
    # ------------------------------------------------------------------------------------------------------------
    reviews_df.to_parquet('data/processed/v2/reviews_engineered.parquet', index=False)
    movies_df.to_parquet('data/processed/v2/movies_engineered.parquet', index=False)
    merged_df.to_parquet('data/processed/v2/merged.parquet', index=False)
    final_df.to_parquet('data/processed/v2/final_engineered.parquet', index=False)

    # # Load Parquet files back into Pandas DataFrames
    # reviews_df = pd.read_parquet('data/processed/reviews_engineered.parquet')
    # movies_df = pd.read_parquet('data/processed/movies_engineered.parquet')
    # merged_df = pd.read_parquet('data/processed/merged.parquet')
    # final_df = pd.read_parquet('data/processed/final_engineered.parquet')

    # -----------------------------
    end_time_single = time.time()
    exec_time_single = end_time_single - start_time_single
    print(f"Execution Time (Single): {exec_time_single:.2f} seconds\n")
    # -----------------------------