import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import datetime
from src.ml_utils import SparseDataset, ConfigurableNN, train, evaluate, train_optuna, get_classification_report

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Take a sample of the data for testing logic
# sample_size = 100000
# X_train = X_train[:sample_size]
# y_train = y_train[:sample_size]
# X_test = X_test[:sample_size]
# y_test = y_test[:sample_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_trial = train_optuna('FF',X_train, y_train, X_test, y_test, input_dim = X_train.shape[1], device = device, n_trials=0, n_epochs=25)

# Extract the best hyperparameters
best_params = best_trial.params
hidden_dim = best_params['hidden_dim']
num_layers = best_params['num_layers']
dropout_rate = best_params['dropout_rate']
lr = best_params['lr']
batch_size = best_params['batch_size']
patience = 10
hidden_dims = [hidden_dim] * (num_layers)*3

# Create DataLoaders for training and test sets
train_dataset = SparseDataset(X_train, y_train)
test_dataset = SparseDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, criterion, optimizer, scheduler, and SummaryWriter
model = ConfigurableNN(X_train.shape[1], hidden_dims, dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr/3)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//3)
writer = SummaryWriter(log_dir=f'runs/optuna/ff_best_trial_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
save_dir ='models/v2/base/pytorch_ff_best_model.pth'

# Train the model with the best hyperparameters
train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, save_dir, patience=patience, epochs=1000)

# Load the saved model
model.load_state_dict(torch.load(save_dir))

# Evaluate the model on the test set and print the classification report
clas_report = get_classification_report(model, test_loader, device)

print(clas_report)
