# Movie Spoiler Detector

## Project Overview

This project aims to detect spoilers in user-generated movie reviews. Using a dataset collected from IMDB, various machine learning models and neural networks are employed to identify reviews that contain spoilers. The project involves extensive data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset

- **Total Records**: 573,913
- **Users**: 263,407
- **Movies**: 1,572
- **Spoiler Reviews**: 150,924
- **Users with at Least One Spoiler Review**: 79,039
- **Items with at Least One Spoiler Review**: 1,570

## Results
TODO
## Future Work
TODO

## Models

### Machine Learning Models

- `adaboost_model`
- `decision_tree_model`
- `gradient_boosting_model`
- `k-nearest_neighbors_model`
- `lightgbm_model`
- `linear_svc_model`
- `logistic_regression_model`
- `random_forest_model`
- `sgd_classifier_model`
- `xgboost_model`

### Best Performing Models (Tuned)

- `lightgbm_model_tuned`
- `xgboost_model_tuned`

### Neural Network Models

- `pytorch_FF_model` (Feedforward Neural Network)
- `pytorch_lstm_model` (LSTM Neural Network)

## Files Description

### `eda.py`

Performs exploratory data analysis (EDA) on the dataset:
- Loading data
- Generating summary statistics
- Plotting distributions of review ratings, spoilers, reviews over time, and movie genres

### `data_preprocessing.py`

Handles preprocessing of the dataset:
- Loading raw data
- Removing duplicates and shuffling data
- Converting date columns to datetime format
- Encoding categorical variables
- Saving processed data to CSV files

### `feature_engineering_nltk_multi.py`

Performs feature engineering:
- Extracting text-based features using NLTK
- Adding date-based features
- Merging review and movie data
- Adding statistical features for movie ratings
- Extracting TF-IDF features and reducing dimensionality with SVD
- Saving engineered datasets to Parquet files

### `feature_engineering_splits.py`

Prepares and splits the data:
- Loads and preprocesses data
- Handles missing values, encodes variables, and scales features
- Applies SMOTE for upsampling and PCA for dimensionality reduction
- Saves data splits and transformation models

### `model_training_base.py`

Train and evaluate various machine learning models on different processed datasets:
- Logistic Regression
- SGD Classifier
- Decision Tree Classifier
- Linear SVC
- Gradient Boosting Classifier
- Random Forest Classifier
- AdaBoost Classifier
- K-Nearest Neighbors Classifier
- XGBoost Classifier
- LightGBM Classifier

**Key Operations**:
1. Load preprocessed data splits
2. Train models
3. Evaluate models and generate classification reports
4. Save trained models

### `model_training_FF.py`

Trains a Feedforward Neural Network (FF):
- Loads train and test datasets
- Trains with Optuna for hyperparameter tuning
- Initializes, trains, and evaluates the model
- Saves the final model and generates a classification report

### `model_training_LSTM.py`

Trains an LSTM Neural Network:
- Similar to `model_training_FF.py` but focuses on LSTM

### `model_tuning_lightgbm.py`

Tunes hyperparameters for the LightGBM model:
- Loads data splits
- Defines parameter grid
- Performs GridSearchCV for hyperparameter tuning
- Trains and evaluates the best model
- Saves the trained model

### `model_tuning_xgboost.py`

Tunes hyperparameters for the XGBoost model:
- Similar to `model_tuning_lightgbm.py` but focuses on XGBoost

### `model_evaluation.py`

Evaluates the performance of the models using various metrics.

### `ml_utils.py`

Contains utilities and classes for training neural networks:
- **Custom Dataset Classes**:
  - `SparseDataset`: Handles sparse matrices
  - `SequenceDataset`: Handles sequence data for LSTM models

- **Model Definitions**:
  - `ConfigurableLSTM`: Configurable LSTM model
  - `ConfigurableNN`: Configurable feed-forward neural network

- **Early Stopping**:
  - `EarlyStopping`: Halts training when validation loss stops improving

- **Training and Evaluation Functions**:
  - `train`: Main training function with early stopping and mixed-precision training
  - `evaluate`: Evaluates the model on the test set
  - `get_classification_report`: Generates classification report

- **Optuna Integration**:
  - `train_optuna`: Integrates Optuna for hyperparameter tuning
  - `objective`: Objective function for Optuna

## ml_utils Example Usage

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import scipy.sparse as sp
import pandas as pd
import torch
from ml_utils import train_optuna

# Example dataset creation
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_sparse = sp.csr_matrix(X)
y = pd.Series(y)

X_train, X_val, y_train, y_val = train_test_split(X_sparse, y, test_size=0.2, random_state=42)

# Train feed-forward neural network using Optuna
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
study = train_optuna('FF', X_train, y_train, X_val, y_val, input_dim=20, device=device, n_trials=50, n_epochs=10)
