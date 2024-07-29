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

df = pd.read_parquet('data/processed/v2/final_engineered.parquet')  # Load the final engineered DataFrame

df['review_summary'] = df['review_summary'].fillna('')  # Fill missing values in 'review_summary' with empty strings
df['plot_synopsis'] = df['plot_synopsis'].fillna('')    # Fill missing values in 'plot_synopsis' with empty strings
df['genre'] = df['genre'].astype('category').cat.codes  # Encode categorical variables as categories

df['is_spoiler'] = df['is_spoiler'].astype(int) # Convert 'is_spoiler' to integer type

X = df.drop(columns=['is_spoiler', 'review_text', 'plot_summary', 
                     'plot_synopsis', 'movie_id', 'review_date', 'release_date'])   # Drop the target and text columns
y = df['is_spoiler']                                                                # Assign the target column to y

print("Initial target class counts:")
print(y.value_counts())


tfidf_vectorizer = TfidfVectorizer(max_features=5000)       # Initialize the TfidfVectorizer
X_text = tfidf_vectorizer.fit_transform(df['review_text'])  # Fit and transform the text data

scaler = StandardScaler()                                               # Initialize the StandardScaler
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))   # Fit and transform the numerical data

X_combined = hstack([X_scaled, X_text]) # Combine the scaled numerical data and the TF-IDF matrix

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42) # Split the data


print("Training set target class counts:")
print(y_train.value_counts())
print("Testing set target class counts:")
print(y_test.value_counts())


joblib.dump(X_train, 'data/processed/v2/splits/base/X_train.pkl')   # Save the training data
joblib.dump(X_test, 'data/processed/v2/splits/base/X_test.pkl')     # Save the testing data
joblib.dump(y_train, 'data/processed/v2/splits/base/y_train.pkl')   # Save the training labels
joblib.dump(y_test, 'data/processed/v2/splits/base/y_test.pkl')     # Save the testing labels
joblib.dump(tfidf_vectorizer, 'models/v2/prep/tfidf_vectorizer.pkl')    # Save the TF-IDF vectorizer
joblib.dump(scaler, 'models/v2/prep/scaler.pkl')                        # Save the StandardScaler


nearest_neighbors = NearestNeighbors(n_jobs=-1)                     # Initialize the NearestNeighbors model
smote = SMOTE(random_state=42, k_neighbors=nearest_neighbors)       # Initialize the SMOTE model with the NearestNeighbors
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) # Fit and transform the training data


print("SMOTE-resampled training set target class counts:")
print(pd.Series(y_train_smote).value_counts())

joblib.dump(X_train_smote, 'data/processed/v2/splits/smote/X_train.pkl')    # Save the SMOTE-resampled training data
joblib.dump(y_train_smote, 'data/processed/v2/splits/smote/y_train.pkl')    # Save the SMOTE-resampled training labels
joblib.dump(X_test, 'data/processed/v2/splits/smote/X_test.pkl')            # Save the testing data
joblib.dump(y_test, 'data/processed/v2/splits/smote/y_test.pkl')            # Save the testing labels

# X_train_smote = joblib.load('data/processed/v2/splits/smote/X_train.pkl')   
# X_test = joblib.load('data/processed/v2/splits/smote/X_test.pkl')
# y_train_smote = joblib.load('data/processed/v2/splits/smote/y_train.pkl')
# y_test = joblib.load('data/processed/v2/splits/smote/y_test.pkl')

initial_pca = IncrementalPCA(n_components=125, batch_size=1000)  # Initialize the IncrementalPCA model

for i in range(0, X_train_smote.shape[0], 1000):            # Fit the PCA model in chunks to avoid memory issues
    end = min(i + 1000, X_train_smote.shape[0])             # Determine the end index of the chunk
    initial_pca.partial_fit(X_train_smote[i:end].toarray()) # Fit the PCA model on the chunk

cumulative_variance_ratio = np.cumsum(initial_pca.explained_variance_ratio_)    # Calculate the cumulative variance ratio
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1              # Determine the number of components for 95% variance

final_pca = IncrementalPCA(n_components=n_components_95, batch_size=1000)   # Initialize the IncrementalPCA model with the determined number of components
for i in range(0, X_train_smote.shape[0], 1000):                            # Fit the PCA model in chunks to avoid memory issues
    end = min(i + 1000, X_train_smote.shape[0])                             # Determine the end index of the chunk
    final_pca.partial_fit(X_train_smote[i:end].toarray())                   # Fit the PCA model on the chunk

X_train_smote_pca = csr_matrix((0, n_components_95))    # Initialize an empty CSR matrix for the PCA-transformed training data
X_test_pca = csr_matrix((0, n_components_95))           # Initialize an empty CSR matrix for the PCA-transformed testing data

# Transform the training data
for i in range(0, X_train_smote.shape[0], 1000):    # Transform the training data in chunks to avoid memory issues
    end = min(i + 1000, X_train_smote.shape[0])     # Determine the end index of the chunk
    X_train_smote_pca = vstack([X_train_smote_pca, final_pca.transform(X_train_smote[i:end].toarray())])    # Transform and stack the chunk

# Transform the testing data
for i in range(0, X_test.shape[0], 1000):           # Transform the testing data in chunks to avoid memory issues
    end = min(i + 1000, X_test.shape[0])            # Determine the end index of the chunk
    X_test_pca = vstack([X_test_pca, final_pca.transform(X_test[i:end].toarray())])   # Transform and stack the chunk


joblib.dump(X_train_smote_pca, 'data/processed/v2/splits/smote_pca/X_train.pkl')    # Save the PCA-transformed training data
joblib.dump(X_test_pca, 'data/processed/v2/splits/smote_pca/X_test.pkl')            # Save the PCA-transformed testing data
joblib.dump(y_train_smote, 'data/processed/v2/splits/smote_pca/y_train.pkl')        # Save the SMOTE-resampled training labels
joblib.dump(y_test, 'data/processed/v2/splits/smote_pca/y_test.pkl')                # Save the testing labels
joblib.dump(final_pca, 'models/v2/prep/pca.pkl')                                    # Save the PCA model