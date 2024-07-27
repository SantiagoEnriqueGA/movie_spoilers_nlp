import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split, HalvingGridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Take a subset of the training data for tuning
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42, stratify=y_train)

# Define the parameter grid for XGBoost
param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'random_state': [42]
}

# Initialize XGBoost classifier
xgb = XGBClassifier(eval_metric='logloss')

# Setup Stratified K-Folds cross-validator
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ----------------------------------------------------
# Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)

# # Perform the grid search
# print("Starting Grid Search for XGBoost on subset...")
# grid_search.fit(X_train_subset, y_train_subset)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters for XGBoost found on subset:")
# print(best_params)

# ----------------------------------------------------
# # Initialize RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, n_iter=30, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2)

# # Perform the random search
# print("Starting Randomized Search for XGBoost on subset...")
# random_search.fit(X_train_subset, y_train_subset)

# # Get the best parameters
# best_params = random_search.best_params_
# print("Best Parameters for XGBoost found on subset:")
# print(best_params)

# ----------------------------------------------------
# Initialize HalvingGridSearchCV
halving_search = HalvingGridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=2, factor=3, min_resources='exhaust', aggressive_elimination=False)

# Perform the halving grid search
print("Starting Halving Grid Search for XGBoost on subset...")
halving_search.fit(X_train_subset, y_train_subset)

# Get the best parameters
best_params = halving_search.best_params_
print("Best Parameters for XGBoost found on subset:")
print(best_params)

# Train the best model on the full training data
best_xgb_full = XGBClassifier(**best_params, eval_metric='error')

print("Training XGBoost with best parameters on full dataset...")
best_xgb_full.fit(X_train, y_train)

# Evaluate the best model
y_pred = best_xgb_full.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print('Best XGBoost Model Classification Report:')
print(report)

# Save the best model
joblib.dump(best_xgb_full, 'models/v2/base/xgboost_tuned_model.pkl')
print('Saved the best XGBoost model trained on full dataset.')
