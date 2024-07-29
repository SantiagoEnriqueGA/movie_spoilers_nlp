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
import time 

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
    reviews_df.shape
    movies_df.shape


    start_time_single = time.time() # Start the timer

    print('Text-Based Features')
    # 1. Text-Based Features 
    # ------------------------------------------------------------------------------------------------------------

    sid = SentimentIntensityAnalyzer() # Initialize the SentimentIntensityAnalyzer

    keywords = ['romance', 'action', 'comedy', 'thriller', 'drama', 
                'horror', 'sci-fi', 'fantasy', 'romantic', 'violence'] # Define keywords to check for

    args = [(review_text, sid, keywords) for review_text in reviews_df['review_text']] # Create a list of arguments for parallel processing

    with mp.Pool(processes=mp.cpu_count()) as pool:     # Use all available CPU cores
        results = pool.map(process_text_features, args) # Process the text features in parallel

    # Unpack the results to DataFrame
    reviews_df['word_count'], reviews_df['avg_word_length'], reviews_df['num_sentences'], reviews_df['compound_sentiment_score'], keyword_flags = zip(*results)

    # Add keyword flags to DataFrame
    for i, keyword in enumerate(keywords):
        reviews_df[f'has_{keyword}'] = [flags[i] for flags in keyword_flags]


    print('Date-Based Features')
    # 2. Date-Based Features
    # ------------------------------------------------------------------------------------------------------------
    reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])   # Convert review_date to datetime
    reviews_df['review_year'] = reviews_df['review_date'].dt.year           # Extract year
    reviews_df['review_month'] = reviews_df['review_date'].dt.month         # Extract month
    reviews_df['review_day'] = reviews_df['review_date'].dt.day             # Extract day


    print('Movie Details Features')
    # 3. Movie Details Features
    # ------------------------------------------------------------------------------------------------------------
    movies_df['duration_minutes'] = movies_df['duration'].apply(convert_duration_to_minutes) # Convert duration to minutes

    # Genre encoding (assuming 'genre' is categorical)
    # movies_df = pd.get_dummies(movies_df, columns=['genre'], prefix='genre')

    merged_df = pd.merge(reviews_df, movies_df, on='movie_id', how='inner') # Merge reviews_df and movies_df on 'movie_id'


    print('Statistical Features')
    # 4. Statistical Features
    # ------------------------------------------------------------------------------------------------------------
    rating_stats = reviews_df.groupby('movie_id')['rating'].agg(['mean', 'median', 'std']).reset_index()    # Calculate rating statistics
    rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']                       # Rename columns

    final_df = pd.merge(merged_df, rating_stats, on='movie_id', how='left') # Merge rating_stats with merged_df on 'movie_id'
    
    
    print('TF-IDF Features')
    # 5. TF-IDF Features
    # ------------------------------------------------------------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=1000)                                              # Initialize the TfidfVectorizer
    tfidf_matrix = tfidf.fit_transform(final_df['review_text'])                             # Fit and transform the text data
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())  # Create a DataFrame from the matrix
    # final_df = pd.concat([final_df, tfidf_df], axis=1)

    svd = TruncatedSVD(n_components=100)                                        # Initialize the TruncatedSVD
    svd_matrix = svd.fit_transform(tfidf_matrix)                                # Fit and transform the TF-IDF matrix
    svd_df = pd.DataFrame(svd_matrix, columns=[f'svd_{i}' for i in range(100)]) # Create a DataFrame from the matrix
    final_df = pd.concat([final_df, svd_df], axis=1)                            # Concatenate the SVD features to the final DataFrame
    

    print('Saving')
    # Save the engineered datasets
    # ------------------------------------------------------------------------------------------------------------
    reviews_df.to_parquet('data/processed/v2/reviews_engineered.parquet', index=False)  # Save reviews_df
    movies_df.to_parquet('data/processed/v2/movies_engineered.parquet', index=False)    # Save movies_df
    merged_df.to_parquet('data/processed/v2/merged.parquet', index=False)               # Save merged_df
    final_df.to_parquet('data/processed/v2/final_engineered.parquet', index=False)      # Save final_df

    # # Load Parquet files back into Pandas DataFrames
    # reviews_df = pd.read_parquet('data/processed/reviews_engineered.parquet')
    # movies_df = pd.read_parquet('data/processed/movies_engineered.parquet')
    # merged_df = pd.read_parquet('data/processed/merged.parquet')
    # final_df = pd.read_parquet('data/processed/final_engineered.parquet')

    # -----------------------------
    end_time_single = time.time()                                       # Stop the timer
    exec_time_single = end_time_single - start_time_single              # Calculate the execution time
    print(f"Execution Time (Single): {exec_time_single:.2f} seconds\n") 
    