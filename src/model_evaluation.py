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

def evaluate_model(model_path, X_test, y_test):
    """
    Evaluates a machine learning model using the provided test data.

    Args:
        model_path (str): The file path to the trained model.
        X_test (array-like): The input features for testing the model.
        y_test (array-like): The true labels for testing the model.

    Returns:
        dict: A dictionary containing the classification report and evaluation time.
    """
    print(f"Evaluating model: {model_path}")    

    model = joblib.load(model_path) # Load the model
    
    start_time = time.time()        # Start the timer
    y_pred = model.predict(X_test)  # Make predictions on the test set
    end_time = time.time()          # Stop the timer

    report = classification_report(y_test, y_pred, output_dict=True)    # Generate the classification report
    report['eval_time'] = end_time - start_time                         # Add the evaluation time to the report
    
    return report

def flatten_report(report, model_name):
    """
    Flattens a nested dictionary report into a single-level dictionary.

    Args:
        report (dict): The nested dictionary report to be flattened.
        model_name (str): The name of the model.

    Returns:
        dict: A single-level dictionary with flattened report and model name.
    """
    flatten = {}
    for key, value in report.items():                   # Iterate over the items in the report
        if isinstance(value, dict):                     
            for sub_key, sub_value in value.items():    # Iterate over the items in the sub-dictionary
                flatten[f"{key}_{sub_key}"] = sub_value # Add the sub-key and sub-value to the flatten dictionary
        else:   
            flatten[key] = value                        # Add the key and value to the flatten dictionary
    flatten['model'] = model_name                       # Add the model name to the flatten dictionary
    
    return flatten


reports = []
for key, path in model_paths.items():           # Iterate over the model paths
    print(f"\nProcessing models in: {path}")    

    for model_file in os.listdir(path):         # Iterate over the files in the model path
        if model_file.endswith('.pkl'):         # Check if the file is a pickle file
            model_name = f"{key}_{model_file.split('_model.pkl')[0]}"   # Extract the model name
            
            # Evaluate the model based on the data type
            if   key == 'base':      report = evaluate_model(os.path.join(path, model_file), X_test_base, y_test_base)  
            elif key == 'smote':     report = evaluate_model(os.path.join(path, model_file), X_test_smote, y_test_smote)
            elif key == 'smote_pca': report = evaluate_model(os.path.join(path, model_file), X_test_smote_pca, y_test_smote_pca)
            
            if report: reports.append(flatten_report(report, model_name)) # Append the flattened report to the reports list

# Evaluate the PyTorch neural network models 
# ---------------------------------------------------------------------------
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')  # Load the training data
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')    # Load the test data
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')  # Load the training labels
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')    # Load the test labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Set the device to GPU if available


# Evaluate the LSTM neural network model
# ---------------------------------------------------------------------------
print("Evaluating LSTM neural network model...")

# Get the best hyperparameters for the LSTM model - n_trials = 0 to load the best model
best_trial_lstm = train_optuna('LSTM',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)

best_params = best_trial_lstm.params        # Extract the best hyperparameters
hidden_dim = best_params['hidden_dim']      # Extract the hidden dimension
num_layers = best_params['num_layers']      # Extract the number of layers
dropout_rate = best_params['dropout_rate']  # Extract the dropout rate
batch_size = best_params['batch_size']      # Extract the batch size

test_dataset = SequenceDataset(X_test, y_test)                                  # Create a Dataset for the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Create a DataLoader for the test set

model_lstm = ConfigurableLSTM(X_train.shape[1], hidden_dim, num_layers, dropout_rate).to(device)    # Initialize the model

save_dir = 'models/v2/base/pytorch_lstm_best_model.pth' # Load the saved model
model_lstm.load_state_dict(torch.load(save_dir))        # Load the model

clas_report = get_classification_report(model_lstm, test_loader, device)    # Get the classification report

lstm_report_flat = flatten_report(clas_report, 'base_pytorch_lstm_best')    # Flatten the LSTM classification report
reports.append(lstm_report_flat)                                            # Add the LSTM report to the reports list


# Evaluate the FF neural network model
# ---------------------------------------------------------------------------
print("Evaluating FF neural network model...")

# Get the best hyperparameters for the FF model - n_trials = 0 to load the best model
best_trial_ff = train_optuna('FF',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)

best_params = best_trial_ff.params          # Extract the best hyperparameters
hidden_dim = best_params['hidden_dim']      # Extract the hidden dimension
num_layers = best_params['num_layers']      # Extract the number of layers
dropout_rate = best_params['dropout_rate']  # Extract the dropout rate
batch_size = best_params['batch_size']      # Extract the batch size
hidden_dims = [hidden_dim] * (num_layers)*3 # Create a list of hidden dimensions

test_dataset = SparseDataset(X_test, y_test)                                    # Create a Dataset for the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Create a DataLoader for the test set

model_ff = ConfigurableNN(X_train.shape[1], hidden_dims, dropout_rate).to(device)   # Initialize the model

save_dir ='models/v2/base/pytorch_ff_best_model.pth'    # Load the saved model
model_ff.load_state_dict(torch.load(save_dir))          # Load the model

clas_report = get_classification_report(model_ff, test_loader, device)  # Get the classification report

ff_report_flat = flatten_report(clas_report, 'base_pytorch_ff_best')    # Flatten the FF classification report
reports.append(ff_report_flat)                                          # Add the FF report to the reports list



# ---------------------------------------------------------------------------
print("Converting reports to DataFrame...")

report_df = pd.DataFrame(reports)

output_file = 'data/processed/evaluation_reports_v2.1.csv'  # Define the output file path
report_df.to_csv(output_file, index=False)                  # Save the report DataFrame to a CSV file
print(f"Evaluation reports saved to '{output_file}'")       # Print the output file path
print(report_df)                                            # Print the report DataFrame    
