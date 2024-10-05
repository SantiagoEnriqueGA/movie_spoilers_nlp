import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------------------------
# Base Data
# ---------------------------------------------------------------------------------------------

X_train = joblib.load('data/processed/v3/splits/base/X_train.pkl')  # Load the training data
X_test = joblib.load('data/processed/v3/splits/base/X_test.pkl')    # Load the testing data
y_train = joblib.load('data/processed/v3/splits/base/y_train.pkl')  # Load the training labels
y_test = joblib.load('data/processed/v3/splits/base/y_test.pkl')    # Load the testing labels

df = pd.read_parquet('data/processed/v3/final_engineered.parquet')  # Load the final engineered DataFrame
df['review_summary'] = df['review_summary'].fillna('')  # Fill missing values in 'review_summary' with empty strings
df['plot_synopsis'] = df['plot_synopsis'].fillna('')    # Fill missing values in 'plot_synopsis' with empty strings
df['genre'] = df['genre'].astype('category').cat.codes  # Encode categorical variables as categories
df['is_spoiler'] = df['is_spoiler'].astype(int) # Convert 'is_spoiler' to integer type

X = df.drop(columns=['user_id','is_spoiler', 'review_text', 'plot_summary', 
                     'plot_synopsis', 'movie_id', 'review_date', 'release_date'])   # Drop the target and text columns
a = X.select_dtypes(include=[np.number])
feature_names = a.columns


# Models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, verbose=0, n_jobs=-1, class_weight='balanced'),
    'SGD Classifier': SGDClassifier(random_state=42, verbose=0, n_jobs=-1, class_weight='balanced'),
    'Linear SVC': LinearSVC(random_state=42, verbose=0, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, verbosity=0, n_jobs=-1, class_weight='balanced'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=0, n_jobs=-1, class_weight='balanced'),
}

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Dictionary to store trained models and classification reports
trained_models = {}
model_reports = {}

# Train and evaluate each model inside the loop
for model_name, model in models.items():
    print(f'Training {model_name}...')
    
    # Use sample weights for XGBoost and LightGBM
    if model_name in ['XGBoost', 'LightGBM']:
        # Train the model with sample weights
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        # Train the model without sample weights
        model.fit(X_train, y_train)
    
    # Store the trained model for later use
    trained_models[model_name] = model
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Store the classification report
    model_reports[model_name] = classification_report(y_test, y_pred)
    
    # Print the classification report
    print(f'Classification Report for {model_name}:')
    print(model_reports[model_name])


# Outside the loop: Extract and print feature importance for each model
for model_name, model in trained_models.items():
    print(f'\nFeature Importance for {model_name}:')

    if model_name in ['Logistic Regression', 'SGD Classifier', 'Linear SVC']:
        # For linear models with coef_
        feature_importance = np.abs(model.coef_).flatten()
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,   # Use the feature_names list
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
        
    elif model_name in ['XGBoost', 'LightGBM']:
        # For tree-based models with feature_importances_
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,   # Use the feature_names list
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)

    print("\n" + "-" * 50 + "\n")
