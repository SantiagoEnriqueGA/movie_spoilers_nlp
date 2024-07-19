import torch
import joblib
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_utils_FF import SparseDataset, ConfigurableNN, train, evaluate

def hyperparameter_search(X_train, y_train, X_test, y_test, param_grid):
    train_dataset = SparseDataset(X_train, y_train)
    test_dataset = SparseDataset(X_test, y_test)

    best_model = None
    best_params = None
    best_score = float('-inf')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    param_list = list(ParameterGrid(param_grid))
    total_params = len(param_list)

    for idx, params in enumerate(param_list):
        batch_size = params['batch_size']
        dropout_rate = params['dropout_rate']
        hidden_dims = params['hidden_dims']
        lr = params['lr']

        print(f"Training model {idx + 1}/{total_params} with parameters: {params}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = ConfigurableNN(X_train.shape[1], hidden_dims, dropout_rate)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        writer = None  # We do not use TensorBoard for hyperparameter search
        
        train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, patience=10, epochs=3)

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_params = params
            best_model = model.state_dict()

    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}")
    return best_model, best_params

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Define the hyperparameter grid
param_grid = {
    'batch_size': [3200, 64, 128],
    'dropout_rate': [0.3, 0.5],
    'hidden_dims': [(64, 128), (128, 256), (256, 512)],  
    'lr': [0.01, 0.001, 0.0001]
}

# Perform hyperparameter search
best_model, best_params = hyperparameter_search(X_train, y_train, X_test, y_test, param_grid)

# Initialize TensorBoard writer with unique name based on current date and time
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(f'runs/pytorch_ffnn_experiment_{timestamp}')

# Log best hyperparameters
writer.add_hparams(best_params, {})

# Create DataLoaders with best parameters
train_dataset = SparseDataset(X_train, y_train)
test_dataset = SparseDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Define the model with best parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConfigurableNN(X_train.shape[1], best_params['hidden_dims'], best_params['dropout_rate'])
model.to(device)
model.load_state_dict(best_model)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Train the model with best hyperparameters
train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, patience=10, epochs=100)

# Evaluate the model
evaluate(model, test_loader, criterion, device)

# Close the TensorBoard writer
writer.close()
