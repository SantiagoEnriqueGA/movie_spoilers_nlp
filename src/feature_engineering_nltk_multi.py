import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import multiprocessing as mp
from datetime import datetime
import time 
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words') 

def process_text_features(args):
    """
    Process text features.

    Args:
        args (tuple): A tuple containing the following elements:
            - review_text (str): The text of the review.
            - sid (SentimentIntensityAnalyzer): An instance of the SentimentIntensityAnalyzer class.
            - keywords (list): A list of keywords to check for in the review text.

    Returns:
        tuple: A tuple containing the following features:
            - word_count (int): The number of words in the review text.
            - avg_word_len (float): The average length of words in the review text.
            - num_sentences (int): The number of sentences in the review text.
            - compound_sentiment (float): The compound sentiment score of the review text.
            - keyword_flags (list): A list of binary flags indicating the presence of keywords in the review text.
    """
    review_text, sid, keywords = args                                                   # Unpack the arguments
    word_count = len(word_tokenize(review_text))                                        # Calculate the word count
    avg_word_len = avg_word_length(review_text)                                         # Calculate the average word length
    num_sentences = len(sent_tokenize(review_text))                                     # Calculate the number of sentences
    compound_sentiment = sid.polarity_scores(review_text)['compound']                   # Calculate the compound sentiment score
    keyword_flags = [int(keyword in review_text.lower()) for keyword in keywords]       # Check for the presence of keywords
    
    return word_count, avg_word_len, num_sentences, compound_sentiment, keyword_flags

def avg_word_length(text):
    """
    Calculate the average word length in a given text.

    Parameters:
    text (str): The input text.

    Returns:
    float: The average word length.
    """
    words = word_tokenize(text)                             # Tokenize the text into words
    return sum(len(word) for word in words) / len(words)    # Calculate the average word length

def convert_duration_to_minutes(duration):
    """
    Converts a duration string in the format 'Xh Ymin' to minutes.

    Parameters:
    duration (str): The duration string to be converted.

    Returns:
    int: The duration in minutes.

    Example:
    convert_duration_to_minutes('2h 30min')  # Returns: 150
    """
    match = re.match(r'(\d+)h (\d+)min', duration)  # Match the duration string
    if match:                                       # If the match is found
        hours = int(match.group(1))                 # Extract the hours
        minutes = int(match.group(2))               # Extract the minutes
        return hours * 60 + minutes                 # Convert to minutes
    else:
        return 0                                    # Return 0 if no match is found

if __name__ == "__main__":
    reviews_df = pd.read_csv('data/processed/reviews.csv')      # Load reviews data
    movies_df = pd.read_csv('data/processed/movie_details.csv') # Load movies data
    # reviews_df.shape
    # movies_df.shape

    total_start_time = time.time() # Start the total timer

    # 1. Text-Based Features 
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Generating Text-Based Features----')
    start_time = time.time() # Start the timer

    sid = SentimentIntensityAnalyzer() # Initialize the SentimentIntensityAnalyzer

    # Spoiler-related keywords
    keywords = [
        'spoiler', 'reveals', 'plot twist', 'ending', 'dies', 'death', 'killer', 
        'murderer', 'secret', 'betrayal', 'identity', 'truth', 'hidden', 'unveiled',
        'climax', 'finale', 'conclusion', 'resolution', 'twist', 'surprise', 
        'unexpected', 'shock', 'disclosure', 'unmask', 'revelation', 'expose', 
        'uncover', 'spoils', 'giveaway', 'leak', 'spoiling', 'foretell', 'foreshadow'
    ]
    args = [(review_text, sid, keywords) for review_text in reviews_df['review_text']] # Create a list of arguments for parallel processing

    with mp.Pool(processes=mp.cpu_count()) as pool:     # Use all available CPU cores
        results = pool.map(process_text_features, args) # Process the text features in parallel

    # Unpack the results to DataFrame
    reviews_df['word_count'], reviews_df['avg_word_length'], reviews_df['num_sentences'], reviews_df['compound_sentiment_score'], keyword_flags = zip(*results)

    # Add keyword flags to DataFrame
    for i, keyword in enumerate(keywords):
        reviews_df[f'has_{keyword}'] = [flags[i] for flags in keyword_flags]

    end_time = time.time()
    print(f"Text-Based Features Execution Time: {end_time - start_time:.2f} seconds\n")
    

    # 2. Date-Based Features
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Generating Date-Based Features----')
    start_time = time.time() # Start the timer
    
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])   # Convert review_date to datetime
    reviews_df['review_year'] = reviews_df['review_date'].dt.year           # Extract year
    reviews_df['review_month'] = reviews_df['review_date'].dt.month         # Extract month
    reviews_df['review_day'] = reviews_df['review_date'].dt.day             # Extract day

    end_time = time.time()
    print(f"Date-Based Features Execution Time: {end_time - start_time:.2f} seconds\n")


    # 3. Movie Details Features
    print('\n----Generating Movie Details Features----')
    # ------------------------------------------------------------------------------------------------------------
    start_time = time.time() # Start the timer
    
    movies_df['duration_minutes'] = movies_df['duration'].apply(convert_duration_to_minutes) # Convert duration to minutes
    # Genre encoding (assuming 'genre' is categorical)
    # movies_df = pd.get_dummies(movies_df, columns=['genre'], prefix='genre')
    merged_df = pd.merge(reviews_df, movies_df, on='movie_id', how='inner') # Merge reviews_df and movies_df on 'movie_id'

    end_time = time.time()
    print(f"Movie Details Features Execution Time: {end_time - start_time:.2f} seconds\n")
    

    # 4. Statistical Features
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Generating Statistical Features----')
    start_time = time.time() # Start the timer
    
    rating_stats = reviews_df.groupby('movie_id')['rating'].agg(['mean', 'median', 'std']).reset_index()    # Calculate rating statistics
    rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']                       # Rename columns

    final_df = pd.merge(merged_df, rating_stats, on='movie_id', how='left') # Merge rating_stats with merged_df on 'movie_id'
    
    end_time = time.time()
    print(f"Statistical Features Execution Time: {end_time - start_time:.2f} seconds\n")
    
    
    # 5. Cosine Similarity Features
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Generating Cosine Similarity Features----')
    start_time = time.time()  # Start the timer

    def compute_cosine_similarity(tfidf_matrix1, tfidf_matrix2):
        return cosine_similarity(tfidf_matrix1, tfidf_matrix2).diagonal()
    
    def compute_cosine_similarity_batched(tfidf_matrix1, tfidf_matrix2, batch_size=100):
        num_samples = tfidf_matrix1.shape[0]
        cosine_similarities = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            # Compute cosine similarity for the current batch
            cosine_sim = cosine_similarity(tfidf_matrix1[start_idx:end_idx], tfidf_matrix2[start_idx:end_idx]).diagonal()
            cosine_similarities.append(cosine_sim)

        # Flatten the list of results into a single array
        return np.hstack(cosine_similarities)

    tfidf_vectorizer = TfidfVectorizer()

    # --- First Pair: Review Text and Plot Summary ---
    # Combine review_text and plot_summary for fitting the vectorizer
    combined_text = final_df['review_text'].fillna('').tolist() + final_df['plot_summary'].fillna('').tolist()

    # Fit the vectorizer on the combined text
    tfidf_vectorizer.fit(combined_text)

    # Transform review_text and plot_summary separately
    tfidf_review_text = tfidf_vectorizer.transform(final_df['review_text'].fillna(''))
    tfidf_plot_summary = tfidf_vectorizer.transform(final_df['plot_summary'].fillna(''))

    # Compute cosine similarity between review_text and plot_summary
    cosine_sim_review_plot = compute_cosine_similarity_batched(tfidf_review_text, tfidf_plot_summary)

    # Add cosine similarity to final_df
    final_df['cosine_sim_review_plot'] = cosine_sim_review_plot

    # --- Second Pair: Review Summary and Plot Synopsis ---
    # Combine review_summary and plot_synopsis for fitting the vectorizer
    combined_summary = final_df['review_summary'].fillna('').tolist() + final_df['plot_synopsis'].fillna('').tolist()

    # Fit the vectorizer on the combined summary
    tfidf_vectorizer.fit(combined_summary)

    # Transform review_summary and plot_synopsis separately
    tfidf_review_summary = tfidf_vectorizer.transform(final_df['review_summary'].fillna(''))
    tfidf_plot_synopsis = tfidf_vectorizer.transform(final_df['plot_synopsis'].fillna(''))

    # Compute cosine similarity between review_summary and plot_synopsis
    cosine_sim_summary_synopsis = compute_cosine_similarity_batched(tfidf_review_summary, tfidf_plot_synopsis)

    # Add cosine similarity to final_df
    final_df['cosine_sim_summary_synopsis'] = cosine_sim_summary_synopsis

    end_time = time.time()
    print(f"Cosine Similarity Features Execution Time: {end_time - start_time:.2f} seconds\n")


    # 6. TF-IDF Features
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Generating TF-IDF Features----')
    start_time = time.time() # Start the timer
    
    tfidf = TfidfVectorizer(max_features=5000)                                                               # Initialize the TfidfVectorizer
    tfidf_matrix = tfidf.fit_transform(final_df['review_text'])                             # Fit and transform the text data
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())  # Create a DataFrame from the matrix
    # final_df = pd.concat([final_df, tfidf_df], axis=1)

    svd = TruncatedSVD(n_components=100)                                            # Initialize the TruncatedSVD
    svd_matrix = svd.fit_transform(tfidf_matrix)                                    # Fit and transform the TF-IDF matrix
    svd_df = pd.DataFrame(svd_matrix, columns=[f'svd_{i}' for i in range(100)])     # Create a DataFrame from the matrix
    final_df = pd.concat([final_df, svd_df], axis=1)                                # Concatenate the SVD features to the final DataFrame

    end_time = time.time()
    print(f"TF-IDF Features Execution Time: {end_time - start_time:.2f} seconds\n")
    
    
    # Save the engineered datasets
    # ------------------------------------------------------------------------------------------------------------
    print('\n----Saving----')
    start_time = time.time() # Start the timer
    
    reviews_df.to_parquet('data/processed/v3/reviews_engineered.parquet', index=False)  # Save reviews_df
    movies_df.to_parquet('data/processed/v3/movies_engineered.parquet', index=False)    # Save movies_df
    merged_df.to_parquet('data/processed/v3/merged.parquet', index=False)               # Save merged_df
    final_df.to_parquet('data/processed/v3/final_engineered.parquet', index=False)      # Save final_df

    # # Load Parquet files back into Pandas DataFrames
    # reviews_df = pd.read_parquet('data/processed/reviews_engineered.parquet')
    # movies_df = pd.read_parquet('data/processed/movies_engineered.parquet')
    # merged_df = pd.read_parquet('data/processed/merged.parquet')
    # final_df = pd.read_parquet('data/processed/final_engineered.parquet')

    end_time = time.time()
    print(f"Saving Execution Time: {end_time - start_time:.2f} seconds\n")

    # -----------------------------
    total_end_time = time.time()  # Stop the total timer
    print(f"\n\nTotal Execution Time: {total_end_time - total_start_time:.2f} seconds")
