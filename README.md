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
### Top 10 Models by Accuracy
|model                        |accuracy|0_precision|0_f1-score|1_precision|1_f1-score|weighted avg_precision|weighted avg_f1-score|eval_time  |
|-----------------------------|--------|-----------|----------|-----------|----------|----------------------|---------------------|-----------|
|base_pytorch_lstm_best       | 0.7934 | 0.8170    | 0.8683   | 0.6746    | 0.5205   | 0.7793               | 0.7762              |25.18037391|
|base_lightgbm_tuned          | 0.7928 | 0.8082    | 0.8699   | 0.7008    | 0.4922   | 0.7798               | 0.7699              |0.438542843|
|base_pytorch_ff_best         | 0.7906 | 0.8190    | 0.8657   | 0.6573    | 0.5245   | 0.7762               | 0.7754              |19.89654732|
|base_lightgbm                | 0.7884 | 0.8036    | 0.8675   | 0.6928    | 0.4740   | 0.7743               | 0.7634              |0.111460686|
|base_xgboost                 | 0.7882 | 0.8060    | 0.8669   | 0.6830    | 0.4824   | 0.7734               | 0.7651              |0.119859934|
|smote_xgboost                | 0.7820 | 0.8050    | 0.8623   | 0.6537    | 0.4767   | 0.7649               | 0.7602              |0.140076637|
|base_linear_svc              | 0.7817 | 0.7988    | 0.8636   | 0.6719    | 0.4535   | 0.7652               | 0.7550              |0.056011677|
|base_logistic_regression     | 0.7815 | 0.8030    | 0.8625   | 0.6572    | 0.4696   | 0.7644               | 0.7585              |0.057740688|
|smote_lightgbm               | 0.7802 | 0.8072    | 0.8604   | 0.6395    | 0.4836   | 0.7628               | 0.7606              |22.3690455 |
|base_gradient_boosting       | 0.7794 | 0.7894    | 0.8642   | 0.6986    | 0.4127   | 0.7654               | 0.7447              |0.46252656 |

### Best Models by Metric
|Metric                |Best Performing Model    |
|----------------------|-------------------------|
|0_precision           |smote_k-nearest_neighbors|
|0_recall              |base_random_forest       |
|0_f1-score            |base_lightgbm_tuned      |
|0_support             |base_adaboost            |
|1_precision           |base_random_forest       |
|1_recall              |smote_k-nearest_neighbors|
|1_f1-score            |smote_logistic_regression|
|1_support             |base_adaboost            |
|accuracy              |base_pytorch_lstm_best   |
|macro avg_precision   |base_xgboost_tuned       |
|macro avg_recall      |smote_logistic_regression|
|macro avg_f1-score    |base_pytorch_ff_best     |
|macro avg_support     |base_adaboost            |
|weighted avg_precision|base_lightgbm_tuned      |
|weighted avg_recall   |base_pytorch_lstm_best   |
|weighted avg_f1-score |base_pytorch_lstm_best   |
|weighted avg_support  |base_adaboost            |
|eval_time             |smote_logistic_regression|

## Class Metrics 

<div style="display: flex; flex-direction: row; justify-content: space-between;">
    <img src="https://github.com/SantiagoEnriqueGA/movie_spoilers_nlp/blob/main/plots/group/class_0_metrics.png" alt="Class 0 Metrics" style="width: 45%;"/>
    <img src="https://github.com/SantiagoEnriqueGA/movie_spoilers_nlp/blob/main/plots/group/class_1_metrics.png" alt="Class 1 Metrics" style="width: 45%;"/>
</div>


### Choices Faced When Selecting the Best Model

1. **Balancing Precision and Recall**:
    - **Precision** measures how many selected items are relevant, while **recall** measures how many relevant items are selected.
    - A model with high precision but low recall (e.g., base_pytorch_lstm_best) might be good for applications where false positives are costly.
    - Conversely, a model with high recall but lower precision (e.g., smote_k-nearest_neighbors) might be suitable where missing a spoiler (false negative) is more detrimental.

2. **Considering Model Complexity and Training Time**:
    - Neural networks like `base_pytorch_lstm_best` and `base_pytorch_ff_best` show good performance but at the cost of higher training and evaluation times.
    - Models like `base_lightgbm_tuned` and `base_xgboost` offer a balance between performance and efficiency, making them attractive choices for practical deployment.

3. **Weighted Metrics**:
    - Weighted metrics consider the support of each class, providing a more balanced view of the model's performance across all data points.
    - For instance, `base_lightgbm_tuned` and `base_pytorch_lstm_best` score high on weighted average metrics, indicating robust overall performance.

4. **Hyperparameter Tuning**:
    - Tuned models (`base_lightgbm_tuned`, `base_xgboost_tuned`) often perform better than their untuned counterparts.
    - Investing time in hyperparameter optimization can yield significant performance gains.

5. **Evaluation Time**:
    - For real-time or near-real-time applications, models with lower evaluation times (e.g., `base_linear_svc`, `base_logistic_regression`) might be preferred despite slightly lower accuracy or F1-scores.
    - `smote_logistic_regression` has the best evaluation time, which might be crucial for high-throughput systems.

7. **Use Case Specific Metrics**:
    - Depending on the end-use case, certain metrics might be prioritized over others. For instance, if avoiding spoilers at any cost is critical, models with higher `1_recall` or `1_f1-score` would be prioritized, despite the increased probability of false positives.

### Conclusion

Selecting the best model for the Movie Spoiler Detector project involves balancing multiple performance metrics, considering model complexity, and evaluating practical constraints like training and evaluation times. The choice ultimately depends on the specific requirements of the deployment environment and the relative importance of precision, recall, and overall efficiency.


## Models

### Machine Learning Models

| Model                  | Description                                                                                        |
|------------------------|----------------------------------------------------------------------------------------------------|
| `adaboost_model`       | Combines multiple weak classifiers to form a strong classifier by adjusting weights on errors.     |
| `decision_tree_model`  | Splits data into subsets based on feature values, forming an intuitive, interpretable tree.         |
| `gradient_boosting_model` | Sequentially builds models to correct errors, optimizing for the loss function using gradient descent. |
| `k-nearest_neighbors_model` | Classifies instances based on the majority class among nearest neighbors in feature space.       |
| `lightgbm_model`       | Efficient gradient boosting framework for large datasets and high-dimensional data.                |
| `linear_svc_model`     | Linear Support Vector Classifier effective in high-dimensional spaces, suitable for text classification. |
| `logistic_regression_model` | Models binary outcomes using the logistic function, simple and interpretable.                     |
| `random_forest_model`  | Ensemble method building multiple decision trees, reducing overfitting and handling large datasets.  |
| `sgd_classifier_model` | Uses stochastic gradient descent to minimize loss, suitable for large-scale learning problems.     |
| `xgboost_model`        | Optimized gradient boosting library, efficient and effective for structured/tabular data.          |

### Best Performing Models (Tuned)

| Model                  | Description                                                                                        |
|------------------------|----------------------------------------------------------------------------------------------------|
| `lightgbm_model_tuned` | LightGBM model with optimized hyperparameters for improved speed and accuracy.                     |
| `xgboost_model_tuned`  | XGBoost model with tuned hyperparameters for maximum performance and efficiency.                   |

### Neural Network Models

| Model                  | Description                                                                                        |
|------------------------|----------------------------------------------------------------------------------------------------|
| `pytorch_FF_model`     | Feedforward neural network trained using backpropagation, used for various tasks.                  |
| `pytorch_lstm_model`   | LSTM neural network effective for sequence prediction, learning long-term dependencies.            |


## Feature Engineering

### Text-Based Features
1. **Word Count**: The total number of words in each review.
2. **Average Word Length**: The average length of words in each review.
3. **Number of Sentences**: The total number of sentences in each review.
4. **Sentiment Score**: The compound sentiment score calculated using the VADER SentimentIntensityAnalyzer.
5. **Keyword Flags**: Binary flags indicating the presence of specific keywords (e.g., "romance", "action", "comedy") in the review text.

### Date-Based Features
1. **Review Year**: The year when the review was written.
2. **Review Month**: The month when the review was written.
3. **Review Day**: The day when the review was written.
4. **Review Day of Week**: The day of the week when the review was written.
5. **Review Time**: The time of the day when the review was written.

### Movie Details Features
1. **Movie Duration**: The duration of the movie, converted to minutes.
2. **Movie Genres**: The genres of the movie, encoded as binary features.

### Statistical Features
1. **Average Rating of Movie**: The average rating of the movie.
2. **Review Length**: The length of the review text.

### TF-IDF Features
1. **TF-IDF Vectors**: Term frequency-inverse document frequency vectors representing the review text.
2. **Dimensionality Reduction**: Applying Truncated Singular Value Decomposition (SVD) to reduce the dimensionality of the TF-IDF vectors.




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
