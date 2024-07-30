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
import optuna

class SparseDataset(Dataset):
    """
    A custom PyTorch dataset for handling sparse data.

    Args:
        X (scipy.sparse.csr_matrix): The input data as a sparse matrix.
        y (pandas.Series): The target labels.

    Returns:
        tuple: A tuple containing the input data and target label as tensors.
    """
    def __init__(self, X, y):   # Initialize the dataset
        self.X = X              # Input data
        self.y = y              # Target labels

    def __len__(self):          # Get the length of the dataset
        return self.X.shape[0]  # Number of samples in the dataset

    def __getitem__(self, idx): # Get a sample from the dataset
        X_data = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()     # Convert sparse matrix to tensor
        y_data = torch.tensor(self.y.iloc[idx], dtype=torch.float32)                    # Convert label to tensor
        return X_data, y_data

class SequenceDataset(Dataset):
    """
    A custom PyTorch dataset for handling sequence data.

    Args:
        X (array-like): The input features.
        y (array-like): The target labels.

    Attributes:
        X (array-like): The input features.
        y (array-like): The target labels.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data at the given index.

    """
    def __init__(self, X, y):   # Initialize the dataset
        self.X = X              # Input data
        self.y = y              # Target labels

    def __len__(self):          # Get the length of the dataset
        return self.X.shape[0]  # Number of samples in the dataset

    def __getitem__(self, idx): # Get a sample from the dataset
        X_data = torch.tensor(self.X[idx].toarray(), dtype=torch.float32)   # Convert sparse matrix to tensor
        y_data = torch.tensor(self.y.iloc[idx], dtype=torch.float32)        # Convert label to tensor
        return X_data, y_data
    
class ConfigurableLSTM(nn.Module):
    """
    Configurable LSTM module.

    Args:
        input_dim (int): The number of expected features in the input x
        hidden_dim (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a `stacked LSTM`, with the second LSTM taking in outputs of the first LSTM and producing the final results. Default: 1
        dropout_rate (float): Dropout rate between LSTM layers. Default: 0.5

    Attributes:
        hidden_dim (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Linear layer for output

    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.5):    # Initialize the module
        super(ConfigurableLSTM, self).__init__()                                # Initialize the parent class
        self.hidden_dim = hidden_dim                                            # Set the hidden dimension
        self.num_layers = num_layers                                            # Set the number of layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout_rate, bidirectional=False)          # LSTM layer
        self.fc = nn.Linear(hidden_dim, 1)                                      # Output layer

    def forward(self, x):
        """
        Forward pass of the LSTM module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size,)
        """
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))   # Forward pass through LSTM
        out = self.fc(out[:, -1, :])        # Use the output from the last time step
        return out.squeeze()                # Return the output
    
class ConfigurableNN(nn.Module):
    """
    A configurable neural network module.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dims (list): A list of integers specifying the number of units in each hidden layer.
        dropout_rate (float, optional): The dropout rate to be applied to the hidden layers. Defaults to 0.5.
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):   # Initialize the module
        super(ConfigurableNN, self).__init__()                      # Initialize the parent class
        layers = []                                                 # Initialize a list to store the layers

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))  # Linear transformation
        layers.append(nn.BatchNorm1d(hidden_dims[0]))        # Batch normalization
        layers.append(nn.ReLU())                             # Activation function
        layers.append(nn.Dropout(dropout_rate))              # Dropout regularization

        for i in range(1, len(hidden_dims)):                            # Iterate over the hidden dimensions
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))  # Linear transformation
            layers.append(nn.BatchNorm1d(hidden_dims[i]))               # Batch normalization
            layers.append(nn.ReLU())                                    # Activation function
            layers.append(nn.Dropout(dropout_rate))                     # Dropout regularization

        layers.append(nn.Linear(hidden_dims[-1], 1))  # Output a single value for BCEWithLogitsLoss

        self.model = nn.Sequential(*layers) # Combine all the layers into a sequential model

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x).squeeze() # Forward pass through the model and return the output
    
class EarlyStopping:
    """
    Class for implementing early stopping during model training.

    Args:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to be considered as improvement.

    Attributes:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to be considered as improvement.
        counter (int): Counter to keep track of epochs without improvement.
        best_loss (float): The best loss achieved so far.
        early_stop (bool): Flag to indicate if early stopping condition is met.

    Methods:
        __call__(self, val_loss, model, save_dir=None): Check if validation loss improved and perform early stopping if necessary.
    """
    def __init__(self, patience=5, delta=0): # Initialize the early stopping object
        self.patience = patience        # Number of epochs to wait for improvement
        self.delta = delta              # Minimum change in validation loss to be considered as improvement
        self.counter = 0                # Counter to keep track of epochs without improvement
        self.best_loss = float('inf')   # Initialize the best loss with infinity
        self.early_stop = False         # Flag to indicate if early stopping condition is met

    def __call__(self, val_loss, model, save_dir=None): # Call the early stopping object
        """
        Check if validation loss improved and perform early stopping if necessary.

        Args:
            val_loss (float): The validation loss.
            model: The model being trained.
            save_dir (str, optional): The directory to save the best model. Defaults to None.
        """
        if val_loss < self.best_loss - self.delta:  # Check if validation loss improved
            self.best_loss = val_loss               # Update the best loss
            self.counter = 0                        # Reset the counter
            if save_dir:            
                # models/v2/base/pytorch_nn_best_model.pth
                # models/v2/base/pytorch_lstm_best_model.pth
                torch.save(model.state_dict(), save_dir) # Save the best model
                print('Model improved and saved.')
        else:
            self.counter += 1                   # Increment the counter
            if self.counter >= self.patience:   # Check if early stopping condition is met
                self.early_stop = True          # Set the early stop flag
                print(f"Early stopping triggered after {self.counter} epochs.")
                                
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, save_dir, patience=5, epochs=10):
    """
    Trains a given model using the specified data loaders, criterion, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer for logging metrics.
        device (torch.device): The device to be used for training.
        save_dir (str): The directory to save the trained model.
        patience (int, optional): The number of epochs to wait for improvement before early stopping. Defaults to 5.
        epochs (int, optional): The number of epochs to train the model. Defaults to 10.
    """
    early_stopping = EarlyStopping(patience)    # Initialize the early stopping object
    scaler = GradScaler()                       # Initialize GradScaler for mixed precision
    
    for epoch in range(epochs): # Iterate over the specified number of epochs
        model.train()           # Set the model to training mode
        running_loss = 0.0      # Initialize the running loss
        correct = 0             # Initialize the number of correct predictions
        total = 0               # Initialize the total number of predictions
        
        # Iterate over the training data
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):  
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            optimizer.zero_grad()                                   # Zero the gradients
            
            with autocast():                        # Mixed precision
                outputs = model(inputs)             # Forward pass
                loss = criterion(outputs, labels)   # Calculate the loss

            scaler.scale(loss).backward()   # Backward pass with scaling
            scaler.step(optimizer)          # Update the weights
            scaler.update()                 # Update the GradScaler
            
            running_loss += loss.item() * inputs.size(0)    # Accumulate the running loss            
            predicted = torch.round(torch.sigmoid(outputs)) # Get the predicted labels
            total += labels.size(0)                         # Increment the total count
            correct += (predicted == labels).sum().item()   # Increment the correct count
        
        epoch_loss = running_loss / len(train_loader.dataset)   # Calculate the average loss per sample
        epoch_accuracy = correct / total                        # Calculate the accuracy
        
        if writer is not None:  # Check if a SummaryWriter is provided
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device) # Evaluate the model on the test set
        
        if writer is not None:  # Check if a SummaryWriter is provided
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        scheduler.step(test_loss)                   # Step the scheduler with the test loss
        early_stopping(test_loss, model, save_dir)  # Call the early stopping function with the test loss and model
        
        if early_stopping.early_stop:   # Check if early stopping condition is met
            break 
            
    get_classification_report(model, test_loader, device) # After training, get classification report

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the performance of a model on the test data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to perform the evaluation on.

    Returns:
        tuple: A tuple containing the test loss and accuracy.
    """
    model.eval()    # Set the model to evaluation mode
    test_loss = 0.0 # Initialize the test loss
    correct = 0     # Initialize the number of correct predictions
    total = 0       # Initialize the total number of predictions

    with torch.no_grad():   # Disable gradient tracking
        for inputs, labels in tqdm(test_loader, desc="Evaluating"): # Iterate over the test data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            
            outputs = model(inputs)                                 # Forward pass
            
            if outputs.shape != labels.shape:   # Check if the shapes are different
                outputs = outputs.view(-1)      # Reshape the outputs
                labels = labels.view(-1)        # Reshape the labels
            
            loss = criterion(outputs, labels)           # Calculate the loss
            test_loss += loss.item() * inputs.size(0)   # Accumulate the test loss
            
            predicted = torch.round(torch.sigmoid(outputs))     # Get the predicted labels
            total += labels.size(0)                             # Increment the total count
            correct += (predicted == labels).sum().item()       # Increment the correct count

    accuracy = correct / total              # Calculate the accuracy
    test_loss /= len(test_loader.dataset)   # Calculate the average test loss per sample

    return test_loss, accuracy  

def get_classification_report(model, test_loader, device):
    """
    Generate a classification report for a given model using the test data.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): The data loader for the test data.
        device (torch.device): The device to run the model on.

    Returns:
        dict: A dictionary containing the classification report metrics.
    """
    model.eval()    # Set the model to evaluation mode
    y_true = []     # Initialize the list to store true labels
    y_pred = []     # Initialize the list to store predicted labels
    
    with torch.no_grad():   # Disable gradient tracking
        for inputs, labels in tqdm(test_loader, desc="Generating Classification Report"):  # Iterate over the test data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            outputs = model(inputs)                                 # Forward pass
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            y_true.extend(labels.cpu().numpy())                     # Append true labels to the list
            y_pred.extend(predicted.cpu().numpy())                  # Append predicted labels to the list
            
    y_true = [int(label) for label in y_true]   # Convert true labels to integers
    y_pred = [int(pred) for pred in y_pred]     # Convert predicted labels to integers
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)
    
def train_optuna(nn_type, X_train, y_train, X_val, y_val, input_dim, device, n_trials=100, n_epochs = 25):
    """
    Train a model using Optuna hyperparameter optimization.

    Args:
        nn_type (str): The type of neural network to train. Possible values are 'LSTM' and 'FF'.
        X_train (torch.Tensor): The training input data.
        y_train (torch.Tensor): The training target data.
        X_val (torch.Tensor): The validation input data.
        y_val (torch.Tensor): The validation target data.
        input_dim (int): The input dimension of the model.
        device (str): The device to train the model on.
        n_trials (int, optional): The number of trials for hyperparameter optimization. Defaults to 100.
        n_epochs (int, optional): The number of epochs to train the model. Defaults to 25.

    Returns:
        optuna.Trial: The best trial found during hyperparameter optimization.
    """

    def objective(trial):
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: An Optuna `Trial` object representing a single trial of the optimization.

        Returns:
            float: The test accuracy achieved by the model with the best hyperparameters found so far.
        """

        if nn_type == 'LSTM':   # Check if the neural network type is LSTM

            hidden_dim = trial.suggest_int('hidden_dim', 64, 1024)          # Hidden dimension
            num_layers = trial.suggest_int('num_layers', 2, 5)              # Number of LSTM layers
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.75)   # Dropout rate
            lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)            # Learning rate
            batch_size = trial.suggest_int('batch_size', 32, 128)           # Batch size
            patience = 5                                                    # Patience for early stopping
            
            if num_layers == 1:     # Adjust dropout rate for single layer LSTM
                dropout_rate = 0.0  # No dropout for single layer LSTM
            
            train_dataset = SequenceDataset(X_train, y_train)   # Create training dataset
            val_dataset = SequenceDataset(X_val, y_val)         # Create validation dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)  # Training DataLoader
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)    # Validation DataLoader
            
            model = ConfigurableLSTM(input_dim, hidden_dim, num_layers, dropout_rate).to(device)    # Initialize the model
            criterion = nn.BCEWithLogitsLoss()                                                      # Loss function
            optimizer = optim.Adam(model.parameters(), lr=lr)                                       # Optimizer
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2)                   # Learning rate scheduler
            writer = SummaryWriter(log_dir=f'runs/optuna_trial_lstm/trial_{trial.number}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')            # TensorBoard writer
            
            early_stopping = EarlyStopping(patience=patience)   # Early stopping object
            scaler = GradScaler()                               # GradScaler for mixed precision

        if nn_type == 'FF':    # Check if the neural network type is FF
            
            hidden_dim = trial.suggest_int('hidden_dim', 64, 1024)          # Hidden dimension
            num_layers = trial.suggest_int('num_layers', 2, 5)              # Number of hidden layers
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.75)   # Dropout rate
            lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)            # Learning rate
            batch_size = trial.suggest_int('batch_size', 32, 128)           # Batch size
            patience = 5                                                    # Patience for early stopping
            hidden_dims = [hidden_dim] * num_layers                         # Hidden dimensions for each layer
            
            train_dataset = SparseDataset(X_train, y_train)     # Create training dataset
            val_dataset = SparseDataset(X_val, y_val)           # Create validation dataset
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)  # Training DataLoader
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  drop_last=True)    # Validation DataLoader
            
            model = ConfigurableNN(input_dim, hidden_dims, dropout_rate).to(device)         # Initialize the model
            criterion = nn.BCEWithLogitsLoss()                                              # Loss function
            optimizer = optim.Adam(model.parameters(), lr=lr)                               # Optimizer
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2)           # Learning rate scheduler
            writer = SummaryWriter(log_dir=f'runs/optuna_trial_ff/trial_{trial.number}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')    # TensorBoard writer
            
            early_stopping = EarlyStopping(patience=patience)   # Early stopping object
            scaler = GradScaler()                               # GradScaler for mixed precision
        

        best_test_accuracy = 0.0    # Initialize the best test accuracy
        
        for epoch in range(n_epochs):   # Iterate over the specified number of epochs
            model.train()               # Set the model to training mode
            running_loss = 0.0          # Initialize the running loss
            correct = 0                 # Initialize the number of correct predictions
            total = 0                   # Initialize the total number of predictions
            
            # Iterate over the training data
            for inputs, labels in tqdm(train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}/{n_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
                optimizer.zero_grad()                                   # Zero the gradients

                with autocast():                        # Mixed precision
                    outputs = model(inputs)             # Forward pass
                    loss = criterion(outputs, labels)   # Calculate the loss

                scaler.scale(loss).backward()           # Backward pass with scaling
                scaler.step(optimizer)                  # Update the weights
                scaler.update()                         # Update the GradScaler

                running_loss += loss.item() * inputs.size(0)    # Accumulate the running loss
                predicted = torch.round(torch.sigmoid(outputs)) # Get the predicted labels
                total += labels.size(0)                         # Increment the total count
                correct += (predicted == labels).sum().item()   # Increment the correct count

            epoch_loss = running_loss / len(train_loader.dataset)   # Calculate the average loss per sample
            epoch_accuracy = correct / total                        # Calculate the accuracy

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)  # Evaluate the model on the test set

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
            
            trial.report(test_accuracy, epoch) # Report the test accuracy to Optuna

            scheduler.step(test_loss)           # Step the scheduler with the test loss
            early_stopping(test_loss, model)    # Call the early stopping function with the test loss and model

            if early_stopping.early_stop or trial.should_prune():   # Check if early stopping condition is met or trial should be pruned
                writer.close()                                      # Close the TensorBoard writer
                return best_test_accuracy                           # Return the best test accuracy
            
            if test_accuracy > best_test_accuracy:                  # Check if the test accuracy improved
                best_test_accuracy = test_accuracy                  # Update the best test accuracy

        writer.close()          # Close the TensorBoard writer
        return test_accuracy    # Return the best test accuracy
    

    study = optuna.create_study(                        # Create an Optuna study
        direction='maximize',                           # Maximize the test accuracy
        storage=f"sqlite:///optuna_study_{nn_type}.db", # SQLite database to store the study
        study_name=f"{nn_type}-hyperparameter-tuning",  # Name of the study
        load_if_exists=True                             # Load the study if it already exists
    )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, gc_after_trial=True, show_progress_bar=False) # Optimize the study
    joblib.dump(study, f"models/v2/optuna_study_{nn_type}.pkl") # Save the study
    
    print("Best trial:")                    
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_trial # Return the best trial