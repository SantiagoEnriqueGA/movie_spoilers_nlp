import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load processed data using dask
client = Client(processes=False)  # Use processes=False for single-machine Dask scheduler
reviews_ddf = dd.read_csv('data/processed/reviews.csv')
movies_ddf = dd.read_csv('data/processed/movie_details.csv')


print('Text-Based Features')
# 1. Text-Based Features 
# ------------------------------------------------------------------------------------------------------------

# Calculate review length
reviews_ddf['review_length'] = reviews_ddf['review_text'].apply(lambda x: len(x), meta=('review_text', 'int'))

# Count number of words
reviews_ddf['word_count'] = reviews_ddf['review_text'].apply(lambda x: len(word_tokenize(x)), meta=('review_text', 'int'))

reviews_ddf.compute()

# Calculate average word length
def avg_word_length(text):
    words = word_tokenize(text)
    return sum(len(word) for word in words) / len(words)

reviews_ddf['avg_word_length'] = reviews_ddf['review_text'].apply(avg_word_length, meta=('review_text', 'float'))

# Count number of sentences
reviews_ddf['num_sentences'] = reviews_ddf['review_text'].apply(lambda x: len(sent_tokenize(x)), meta=('review_text', 'int'))

# Example of presence of keywords
keywords = ['romance', 'action', 'comedy', 'thriller', 'drama', 'horror', 'sci-fi', 'fantasy', 'romantic', 'violence']
for keyword in keywords:
    reviews_ddf[f'has_{keyword}'] = reviews_ddf['review_text'].str.contains(keyword, case=False, na=False, regex=False).astype(int)

# Example of sentiment analysis using Vader
sid = SentimentIntensityAnalyzer()
reviews_ddf['compound_sentiment_score'] = reviews_ddf['review_text'].apply(lambda x: sid.polarity_scores(x)['compound'], meta=('review_text', 'float'))

# Example of named entity recognition (NER) features
# def count_named_entities(text):
#     return len(nltk.ne_chunk(nltk.pos_tag(word_tokenize(text))))
# reviews_ddf['num_named_entities'] = reviews_ddf['review_text'].apply(count_named_entities, meta=('review_text', 'int'))


print('Date-Based Features')
# 2. Date-Based Features
# ------------------------------------------------------------------------------------------------------------

reviews_ddf['review_year'] = dd.to_datetime(reviews_ddf['review_date']).dt.year

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
movies_ddf['duration_minutes'] = movies_ddf['duration'].apply(convert_duration_to_minutes, meta=('duration', 'int'))

# Convert 'genre' column to categorical dtype
movies_ddf['genre'] = movies_ddf['genre'].astype('category')

# Use dask's `get_dummies` after categorizing 'genre'
movies_ddf = dd.get_dummies(movies_ddf.categorize(), columns=['genre'], prefix='genre')

# Merge reviews_df and movies_df if necessary based on 'movie_id'
merged_ddf = dd.merge(reviews_ddf, movies_ddf, on='movie_id', how='inner')


print('Statistical Features')
# 4. Statistical Features
# ------------------------------------------------------------------------------------------------------------

rating_stats = reviews_ddf.groupby('movie_id')['rating'].agg(['mean', 'median', 'std'], shuffle='tasks').reset_index()
rating_stats.columns = ['movie_id', 'rating_mean', 'rating_median', 'rating_std']

# Merge rating_stats with merged_df on 'movie_id'
final_ddf = dd.merge(merged_ddf, rating_stats, on='movie_id', how='left')


print('Saving')
# Save the engineered datasets
# ------------------------------------------------------------------------------------------------------------
reviews_ddf.compute().to_csv('data/processed/reviews_engineered.csv', index=False)
movies_ddf.compute().to_csv('data/processed/movies_engineered.csv', index=False)
merged_ddf.compute().to_csv('data/processed/merged.csv', index=False)
final_ddf.compute().to_csv('data/processed/final_engineered.csv', index=False)

# Close the dask client
client.close()