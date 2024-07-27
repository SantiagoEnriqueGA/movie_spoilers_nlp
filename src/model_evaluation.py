import joblib
import pandas as pd
from sklearn.metrics import classification_report
import os
import time
import torch
from torch.utils.data import DataLoader
from src.ml_utils import SequenceDataset, ConfigurableLSTM, ConfigurableNN, SparseDataset, train_optuna, get_classification_report
import threading

import warnings
warnings.filterwarnings("ignore")

# Load the datasets
print("Loading datasets...")
X_test_base = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_test_base = joblib.load('data/processed/v2/splits/base/y_test.pkl')

X_test_smote = joblib.load('data/processed/v2/splits/smote/X_test.pkl')
y_test_smote = joblib.load('data/processed/v2/splits/smote/y_test.pkl')

X_test_smote_pca = joblib.load('data/processed/v2/splits/smote_pca/X_test.pkl')
y_test_smote_pca = joblib.load('data/processed/v2/splits/smote_pca/y_test.pkl')

# Define model paths
model_paths = {
    'base': 'models/v2/base/',
    'smote': 'models/v2/smote/',
    'smote_pca': 'models/v2/smote_pca/'
}

# # Define a function to load and evaluate models with a timeout
# def evaluate_model(model_path, X_test, y_test, time_limit=300):
#     print(f"Evaluating model: {model_path}")
#     model = joblib.load(model_path)
#     y_pred = []

#     def predict():
#         nonlocal y_pred
#         y_pred = model.predict(X_test)

#     thread = threading.Thread(target=predict)
#     start_time = time.time()
#     thread.start()
#     thread.join(timeout=time_limit)
#     end_time = time.time()

#     if thread.is_alive():
#         print(f"Model {model_path} took too long to evaluate. Skipping...")
#         return None

#     report = classification_report(y_test, y_pred, output_dict=True)
#     report['eval_time'] = end_time - start_time
#     return report

# Define a function to load and evaluate models
def evaluate_model(model_path, X_test, y_test):
    print(f"Evaluating model: {model_path}")
    model = joblib.load(model_path)
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    report = classification_report(y_test, y_pred, output_dict=True)
    report['eval_time'] = end_time - start_time
    return report

# Define a function to flatten reports into a DataFrame
def flatten_report(report, model_name):
    flatten = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flatten[f"{key}_{sub_key}"] = sub_value
        else:
            flatten[key] = value
    flatten['model'] = model_name
    return flatten

# Load and evaluate sklearn models
reports = []
for key, path in model_paths.items():
    print(f"\nProcessing models in: {path}")
    for model_file in os.listdir(path):
        if model_file.endswith('.pkl'):
            model_name = f"{key}_{model_file.split('_model.pkl')[0]}"
            if key == 'base':
                report = evaluate_model(os.path.join(path, model_file), X_test_base, y_test_base)
            elif key == 'smote':
                report = evaluate_model(os.path.join(path, model_file), X_test_smote, y_test_smote)
            elif key == 'smote_pca':
                report = evaluate_model(os.path.join(path, model_file), X_test_smote_pca, y_test_smote_pca)
            
            if report:
                reports.append(flatten_report(report, model_name))

# ---------------------------------------------------------------------------

# Load the data splits for the NN models
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Evaluate the neural network model
print("Evaluating LSTM neural network model...")

best_trial_lstm = train_optuna('LSTM',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)
# Extract the best hyperparameters
best_params = best_trial_lstm.params
hidden_dim = best_params['hidden_dim']
num_layers = best_params['num_layers']
dropout_rate = best_params['dropout_rate']
batch_size = best_params['batch_size']

# Create DataLoader for the test set
test_dataset = SequenceDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model_lstm = ConfigurableLSTM(X_train.shape[1], hidden_dim, num_layers, dropout_rate).to(device)

# Load the saved model
save_dir = 'models/v2/base/pytorch_lstm_best_model.pth'
model_lstm.load_state_dict(torch.load(save_dir))

# Evaluate the model on the test set and get the classification report
clas_report = get_classification_report(model_lstm, test_loader, device)

# Flatten the NN classification report and add it to the reports list
lstm_report_flat = flatten_report(clas_report, 'base_pytorch_lstm_best')
reports.append(lstm_report_flat)

# ---------------------------------------------------------------------------

# Evaluate the neural network model
print("Evaluating FF neural network model...")

best_trial_ff = train_optuna('FF',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)

# Extract the best hyperparameters
best_params = best_trial_ff.params
hidden_dim = best_params['hidden_dim']
num_layers = best_params['num_layers']
dropout_rate = best_params['dropout_rate']
batch_size = best_params['batch_size']
hidden_dims = [hidden_dim] * (num_layers)*3

# Create DataLoader for the test set
test_dataset = SparseDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model_ff = ConfigurableNN(X_train.shape[1], hidden_dims, dropout_rate).to(device)

# Load the saved model
save_dir ='models/v2/base/pytorch_ff_best_model.pth'
model_ff.load_state_dict(torch.load(save_dir))

# Evaluate the model on the test set and get the classification report
clas_report = get_classification_report(model_ff, test_loader, device)

# Flatten the NN classification report and add it to the reports list
ff_report_flat = flatten_report(clas_report, 'base_pytorch_ff_best')
reports.append(ff_report_flat)
# ---------------------------------------------------------------------------

# Convert the list of flattened reports to a DataFrame
print("Converting reports to DataFrame...")
report_df = pd.DataFrame(reports)

# Save the DataFrame to a CSV file
output_file = 'data/processed/evaluation_reports_v2.1.csv'
report_df.to_csv(output_file, index=False)
print(f"Evaluation reports saved to '{output_file}'")

# Load and print the report file
print("Loading and printing the evaluation report...")
loaded_report_df = pd.read_csv(output_file)
print(loaded_report_df)
