import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, vstack
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

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
joblib.dump(X_train, 'data/processed/splits/base/X_train.pkl')
joblib.dump(X_test, 'data/processed/splits/base/X_test.pkl')
joblib.dump(y_train, 'data/processed/splits/base/y_train.pkl')
joblib.dump(y_test, 'data/processed/splits/base/y_test.pkl')

# Save the vectorizer and scaler
joblib.dump(tfidf_vectorizer, 'models/prep/tfidf_vectorizer.pkl')
joblib.dump(scaler, 'models/prep/scaler.pkl')

# SMOTE Upsampling
nearest_neighbors = NearestNeighbors(n_jobs=-1)
smote = SMOTE(random_state=42, k_neighbors=nearest_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Save SMOTE splits
joblib.dump(X_train_smote, 'data/processed/splits/smote/X_train.pkl')
joblib.dump(y_train_smote, 'data/processed/splits/smote/y_train.pkl')
joblib.dump(X_test, 'data/processed/splits/smote/X_test.pkl')
joblib.dump(y_test, 'data/processed/splits/smote/y_test.pkl')

X_train_smote = joblib.load('data/processed/splits/smote/X_train.pkl')
X_test = joblib.load('data/processed/splits/smote/X_test.pkl')
y_train_smote = joblib.load('data/processed/splits/smote/y_train.pkl')
y_test = joblib.load('data/processed/splits/smote/y_test.pkl')

# PCA for Dimensionality Reduction using 95% explained variance heuristic
initial_pca = IncrementalPCA(n_components=125, batch_size=1000)  # Fit with a batch size equal to the initial number of components

# Fit IncrementalPCA in chunks to avoid memory issues
for i in range(0, X_train_smote.shape[0], 1000):
    end = min(i + 1000, X_train_smote.shape[0])
    initial_pca.partial_fit(X_train_smote[i:end].toarray())

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(initial_pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

# Fit IncrementalPCA with the determined number of components
final_pca = IncrementalPCA(n_components=n_components_95, batch_size=1000)
for i in range(0, X_train_smote.shape[0], 1000):
    end = min(i + 1000, X_train_smote.shape[0])
    final_pca.partial_fit(X_train_smote[i:end].toarray())

# Transform the training and test data using the fitted PCA in chunks to avoid memory issues
X_train_smote_pca = csr_matrix((0, n_components_95))
X_test_pca = csr_matrix((0, n_components_95))

for i in range(0, X_train_smote.shape[0], 1000):
    end = min(i + 1000, X_train_smote.shape[0])
    X_train_smote_pca = vstack([X_train_smote_pca, final_pca.transform(X_train_smote[i:end].toarray())])

for i in range(0, X_test.shape[0], 1000):
    end = min(i + 1000, X_test.shape[0])
    X_test_pca = vstack([X_test_pca, final_pca.transform(X_test[i:end].toarray())])


# Save PCA splits
joblib.dump(X_train_smote_pca, 'data/processed/splits/smote_pca/X_train.pkl')
joblib.dump(X_test_pca, 'data/processed/splits/smote_pca/X_test.pkl')
joblib.dump(y_train_smote, 'data/processed/splits/smote_pca/y_train.pkl')
joblib.dump(y_test, 'data/processed/splits/smote_pca/y_test.pkl')

# Save PCA model
joblib.dump(final_pca, 'models/prep/pca.pkl')