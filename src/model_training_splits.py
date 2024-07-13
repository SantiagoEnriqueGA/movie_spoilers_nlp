import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

# Load the merged data
df = pd.read_parquet('data/processed/final_engineered.parquet')

# Handle missing values
df['review_summary'] = df['review_summary'].fillna('')
df['plot_synopsis'] = df['plot_synopsis'].fillna('')

# Encode categorical variables (e.g., 'genre')
df['genre'] = df['genre'].astype('category').cat.codes

# Convert 'is_spoiler' to int
df['is_spoiler'] = df['is_spoiler'].astype(int)

# Define features and target variable
X = df.drop(columns=['is_spoiler', 'review_text', 'plot_summary', 'plot_synopsis', 'movie_id', 'review_date', 'release_date'])
y = df['is_spoiler']

# Vectorize the 'review_text' column
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_text = tfidf_vectorizer.fit_transform(df['review_text'])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

# Combine numerical and text features
X_combined = hstack([X_scaled, X_text])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Save splits
joblib.dump(X_train, 'data/processed/splits/X_train.pkl')
joblib.dump(X_test, 'data/processed/splits/X_test.pkl')
joblib.dump(y_train, 'data/processed/splits/y_train.pkl')
joblib.dump(y_test, 'data/processed/splits/y_test.pkl')

# Save the vectorizer and scaler
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')