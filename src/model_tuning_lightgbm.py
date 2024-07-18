import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Take a subset of the training data for tuning
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42, stratify=y_train)

# Define the parameter grid for LightGBM
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [20, 30, 40],
    'boosting_type': ['gbdt'],
    'objective': ['binary'],
    'random_state': [42]
}

# Initialize LightGBM classifier
lgbm = LGBMClassifier(force_col_wise=True)

# Setup Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)

# Perform the grid search
print("Starting Grid Search for LightGBM on subset...")
grid_search.fit(X_train_subset, y_train_subset)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters for LightGBM found on subset:")
print(best_params)

# Train the best model on the full training data
best_lgbm_full = LGBMClassifier(**best_params)

print("Training LightGBM with best parameters on full dataset...")
best_lgbm_full.fit(X_train, y_train)

# Evaluate the best model
y_pred = best_lgbm_full.predict(X_test)
report = classification_report(y_test, y_pred)
print('Best LightGBM Model Classification Report:')
print(report)

# Save the best model
joblib.dump(best_lgbm_full, 'models/v2/base/lightgbm_model_tuned.pkl')
print('Saved the best LightGBM model trained on full dataset.')
