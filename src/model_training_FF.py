import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import datetime
from src.ml_utils import SparseDataset, ConfigurableNN, train, evaluate, train_optuna, get_classification_report

X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')  # Load the training data
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')    # Load the test data
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')  # Load the training labels
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')    # Load the test labels

# Take a sample of the data for testing logic
# sample_size = 100000
# X_train = X_train[:sample_size]
# y_train = y_train[:sample_size]
# X_test = X_test[:sample_size]
# y_test = y_test[:sample_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Check if GPU is available

# Perform hyperparameter tuning using Optuna
best_trial = train_optuna('FF',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)

best_params = best_trial.params             # Extract the best hyperparameters
hidden_dim = best_params['hidden_dim']      # Extract the hidden dimension
num_layers = best_params['num_layers']      # Extract the number of layers
dropout_rate = best_params['dropout_rate']  # Extract the dropout rate
lr = best_params['lr']                      # Extract the learning rate
batch_size = best_params['batch_size']      # Extract the batch size
patience = 10                               # Set the patience for the ReduceLROnPlateau scheduler
hidden_dims = [hidden_dim] * (num_layers)*3 # Create a list of hidden dimensions

train_dataset = SparseDataset(X_train, y_train)                                 # Create a Dataset for the training set
test_dataset = SparseDataset(X_test, y_test)                                    # Create a Dataset for the test set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # Create a DataLoader for the training set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # Create a DataLoader for the test set


model = ConfigurableNN(X_train.shape[1], hidden_dims, dropout_rate).to(device)  # Initialize the model
criterion = nn.BCEWithLogitsLoss()                                              # Initialize the loss function
optimizer = optim.Adam(model.parameters(), lr=lr/3)                             # Initialize the optimizer
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//3)           # Initialize the scheduler
writer = SummaryWriter(log_dir=f'runs/optuna/ff_best_trial_
                       {datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')    # Initialize the TensorBoard writer
save_dir ='models/v2/base/pytorch_ff_best_model.pth'                            # Set the save directory

# Train the model with the best hyperparameters
train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, save_dir, patience=patience, epochs=1000)
model.load_state_dict(torch.load(save_dir)) # Load the best model

clas_report = get_classification_report(model, test_loader, device) # Get the classification report
print(clas_report)  
