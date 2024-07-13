import pandas as pd
import re
import spacy
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Load processed data
reviews_df = pd.read_csv('data/processed/reviews.csv').iloc[:1000]
movies_df = pd.read_csv('data/processed/movie_details.csv')
reviews_df.shape
movies_df.shape

# -----------------------------
import time
start_time_single = time.time()
# -----------------------------

# Initialize SpaCy with English model and add sentencizer
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

print('Text-Based Features')
# 1. Text-Based Features 
# ------------------------------------------------------------------------------------------------------------

# Function to process text in batches
def process_text_batch(texts):
    docs = list(nlp.pipe(texts))
    return docs

# Calculate review length
reviews_df['review_length'] = reviews_df['review_text'].apply(len)

# Batch process text for word count and average word length
text_docs = process_text_batch(reviews_df['review_text'].tolist())
reviews_df['word_count'] = [len(doc) for doc in text_docs]
reviews_df['avg_word_length'] = [sum(len(token.text) for token in doc) / len(doc) for doc in text_docs]


# Batch process text for number of sentences
reviews_df['num_sentences'] = [len(list(doc.sents)) for doc in text_docs]

# Example of presence of keywords
keywords = ['romance', 'action', 'comedy', 'thriller', 'drama', 'horror', 'sci-fi', 'fantasy', 'romantic', 'violence']
for keyword in keywords:
    reviews_df[f'has_{keyword}'] = reviews_df['review_text'].str.contains(keyword, case=False).astype(int)

# # Example of sentiment analysis using Vader
sid = SentimentIntensityAnalyzer()
reviews_df['compound_sentiment_score'] = reviews_df['review_text'].apply(lambda x: sid.polarity_scores(x)['compound'])


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