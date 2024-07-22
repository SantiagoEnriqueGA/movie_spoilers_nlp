import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from torch.cuda.amp import GradScaler, autocast

# Custom Dataset class to handle sparse matrices
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_data = torch.tensor(self.X[idx].toarray(), dtype=torch.float32)  # Ensure correct shape
        y_data = torch.tensor(self.y.iloc[idx], dtype=torch.float32)  
        return X_data, y_data

# Define the configurable LSTM model
class ConfigurableLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super(ConfigurableLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)  # Output a single value for BCEWithLogitsLoss

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))  # Forward pass through LSTM
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out.squeeze()

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience        # Number of epochs to wait for improvement
        self.delta = delta              # Minimum change in validation loss to be considered as improvement
        self.counter = 0                # Counter to keep track of epochs without improvement
        self.best_loss = float('inf')   # Initialize the best loss with infinity
        self.early_stop = False         # Flag to indicate if early stopping condition is met

    def __call__(self, val_loss, model, save = True):
        if val_loss < self.best_loss - self.delta:  # Check if validation loss improved
            self.best_loss = val_loss               # Update the best loss
            self.counter = 0                        # Reset the counter
            if save == True:
                torch.save(model.state_dict(), 'models/v2/base/pytorch_lstm_best_model.pth')  # Save the model
                print('Model improved and saved.')
        else:
            self.counter += 1                   # Increment the counter
            if self.counter >= self.patience:   # Check if early stopping condition is met
                self.early_stop = True          # Set the early stop flag
                print(f"Early stopping triggered after {self.counter} epochs.")

# Training function with early stopping
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, patience=5, epochs=10):
    early_stopping = EarlyStopping(patience)    # Initialize the early stopping object
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    
    for epoch in range(epochs): # Iterate over the specified number of epochs
        model.train()           # Set the model to training mode
        running_loss = 0.0      # Initialize the running loss
        correct = 0             # Initialize the number of correct predictions
        total = 0               # Initialize the total number of predictions
        
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):  # Iterate over the training data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            optimizer.zero_grad()                                   # Zero the gradients
            
            with autocast():  # Mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)            # Accumulate the running loss            
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            total += labels.size(0)                                 # Increment the total count
            correct += (predicted == labels).sum().item()           # Increment the correct count
        
        epoch_loss = running_loss / len(train_loader.dataset)   # Calculate the average loss per sample
        epoch_accuracy = correct / total                        # Calculate the accuracy
        
        if writer is not None:
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Evaluate on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        if writer is not None:
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Step the scheduler with the test loss
        scheduler.step(test_loss) # Use the validation/test loss for ReduceLROnPlateau
        
        early_stopping(test_loss, model) # Call the early stopping function with the test loss and model
        
        if early_stopping.early_stop:   # Check if early stopping condition is met
            break 
            
    # After training, get classification report
    get_classification_report(model, test_loader, device)

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()    # Set the model to evaluation mode
    test_loss = 0.0 # Initialize the test loss
    correct = 0     # Initialize the number of correct predictions
    total = 0       # Initialize the total number of predictions

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"): # Iterate over the test data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            
            outputs = model(inputs)                                 # Forward pass
            
            # Reshape outputs and labels to the same shape if needed
            if outputs.shape != labels.shape:
                outputs = outputs.view(-1)
                labels = labels.view(-1)
            
            loss = criterion(outputs, labels)                       # Calculate the loss
            test_loss += loss.item() * inputs.size(0)               # Accumulate the test loss
            
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            total += labels.size(0)                                 # Increment the total count
            correct += (predicted == labels).sum().item()           # Increment the correct count

    accuracy = correct / total              # Calculate the accuracy
    test_loss /= len(test_loader.dataset)   # Calculate the average test loss per sample

    return test_loss, accuracy

# Function to get classification report
def get_classification_report(model, test_loader, device):
    model.eval()    # Set the model to evaluation mode
    y_true = []     # Initialize the list to store true labels
    y_pred = []     # Initialize the list to store predicted labels
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Generating Classification Report"):  # Iterate over the test data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            outputs = model(inputs)                                 # Forward pass
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            y_true.extend(labels.cpu().numpy())                     # Append true labels to the list
            y_pred.extend(predicted.cpu().numpy())                  # Append predicted labels to the list
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

import optuna
def train_optuna(X_train, y_train, X_val, y_val, input_dim, device, n_trials=100, n_epochs = 25):
    def objective(trial):
        # Define the hyperparameter search space
        hidden_dim = trial.suggest_int('hidden_dim', 64, 1024)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.75)
        lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True) 
        batch_size = trial.suggest_int('batch_size', 32, 128)
        patience = 5
        
        # Adjust dropout rate if num_layers is 1
        if num_layers == 1:
            dropout_rate = 0.0
        
        # Create DataLoaders
        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model, criterion, optimizer, scheduler, and SummaryWriter
        model = ConfigurableLSTM(input_dim, hidden_dim, num_layers, dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2)  
        writer = SummaryWriter(log_dir=f'runs/optuna_trial_lstm/trial_{trial.number}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
        # Training loop
        early_stopping = EarlyStopping(patience=patience)
        scaler = GradScaler()
        
        best_test_accuracy = 0.0
        
        # Train the model
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}/{n_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast():  # Mixed precision
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct / total

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            # Evaluate on the test set
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
            
            # Add test accuracy to Optuna's report
            trial.report(test_accuracy, epoch)

            # Step the scheduler with the test loss
            scheduler.step(test_loss)

            early_stopping(test_loss, model, save = False)

            if early_stopping.early_stop or trial.should_prune():
                writer.close()
                # Mark trial as complete by returning the best test accuracy so far
                return best_test_accuracy
            
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy

        writer.close()
        return test_accuracy

    # Define the Optuna study with storage URL
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///optuna_study_LSTM.db",  # Specify the storage URL here.
        study_name="lstm-hyperparameter-tuning",
        load_if_exists=True  # Load the study if it already exists
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, gc_after_trial=True, show_progress_bar=True)
    
    # Save the study results
    joblib.dump(study, f"models/v2/optuna_study_LSTM.pkl")
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial

    # Usage
    # best_trial = train_optuna(X_train, y_train, X_val, y_val, input_dim, device)

