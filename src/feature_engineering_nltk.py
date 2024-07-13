import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')


# Load processed data
reviews_df = pd.read_csv('data/processed/reviews.csv').iloc[:10000]
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

# Calculate review length
reviews_df['review_length'] = reviews_df['review_text'].str.len()

# Count number of words
reviews_df['word_count'] = reviews_df['review_text'].apply(lambda x: len(word_tokenize(x)))

# Calculate average word length
def avg_word_length(text):
    words = word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

reviews_df['avg_word_length'] = reviews_df['review_text'].apply(avg_word_length)

# Count number of sentences
reviews_df['num_sentences'] = reviews_df['review_text'].apply(lambda x: len(sent_tokenize(x)))

# Example of presence of keywords
keywords = ['romance', 'action', 'comedy', 'thriller', 'drama', 'horror', 'sci-fi', 'fantasy', 'romantic', 'violence']
for keyword in keywords:
    reviews_df[f'has_{keyword}'] = reviews_df['review_text'].str.contains(keyword, case=False).astype(int)

# Example of sentiment analysis using Vader
sid = SentimentIntensityAnalyzer()
reviews_df['compound_sentiment_score'] = reviews_df['review_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Example of named entity recognition (NER) features
# def count_named_entities(text):
#     return len(nltk.ne_chunk(nltk.pos_tag(word_tokenize(text))))
# reviews_df['num_named_entities'] = reviews_df['review_text'].apply(count_named_entities)


print('Date-Based Features')
# 2. Date-Based Features
# ------------------------------------------------------------------------------------------------------------

reviews_df['review_year'] = pd.to_datetime(reviews_df['review_date']).dt.year

print('Movie Details Features')
# 3. Movie Details Features
# ------------------------------------------------------------------------------------------------------------

# Function to convert duration to minutes
def convert_duration_to_minutes(duration):
    match = re.match(r'(\d+)h (\d+)min', duration)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
    else:
        return 0

# Duration feature (assuming it's available in movie_details.csv)
movies_df['duration_minutes'] = movies_df['duration'].apply(convert_duration_to_minutes)

# Genre encoding (assuming 'genre' is categorical)
movies_df = pd.get_dummies(movies_df, columns=['genre'], prefix='genre')

# Merge reviews_df and movies_df if necessary based on 'movie_id'
merged_df = pd.merge(reviews_df, movies_df, on='movie_id', how='inner')


print('Statistical Features')
# 4. Statistical Features
# ------------------------------------------------------------------------------------------------------------

rating_stats = reviews_df.groupby('movie_id')['rating'].agg(['mean', 'median', 'std']).reset_index()
rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']

# Merge rating_stats with merged_df on 'movie_id'
final_df = pd.merge(merged_df, rating_stats, on='movie_id', how='left')


print('Saving')
# Save the engineered datasets
# ------------------------------------------------------------------------------------------------------------
reviews_df.to_parquet('data/processed/reviews_engineered.parquet', index=False)
movies_df.to_parquet('data/processed/movies_engineered.parquet', index=False)
merged_df.to_parquet('data/processed/merged.parquet', index=False)
final_df.to_parquet('data/processed/final_engineered.parquet', index=False)

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