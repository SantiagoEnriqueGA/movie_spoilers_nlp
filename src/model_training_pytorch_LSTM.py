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

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

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

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
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

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:  # Check if validation loss improved
            self.best_loss = val_loss               # Update the best loss
            self.counter = 0                        # Reset the counter
            torch.save(model.state_dict(), 'models/v2/base/pytorch_lstm_best_model.pth')  # Save the model
            print('Model improved and saved.')
        else:
            self.counter += 1                   # Increment the counter
            if self.counter >= self.patience:   # Check if early stopping condition is met
                self.early_stop = True          # Set the early stop flag
                print(f"Early stopping triggered after {self.counter} epochs.")

# Training function with early stopping
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, patience=5, epochs=10):
    early_stopping = EarlyStopping(patience)    # Initialize the early stopping object
    
    for epoch in range(epochs): # Iterate over the specified number of epochs
        model.train()           # Set the model to training mode
        running_loss = 0.0      # Initialize the running loss
        correct = 0             # Initialize the number of correct predictions
        total = 0               # Initialize the total number of predictions
        
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):  # Iterate over the training data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            optimizer.zero_grad()                                   # Zero the gradients
            outputs = model(inputs)                                 # Forward pass
            loss = criterion(outputs, labels)                       # Calculate the loss
            loss.backward()                                         # Backward pass
            optimizer.step()                                        # Update the weights
            running_loss += loss.item() * inputs.size(0)            # Accumulate the running loss
            
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            total += labels.size(0)                                 # Increment the total count
            correct += (predicted == labels).sum().item()           # Increment the correct count
        
        epoch_loss = running_loss / len(train_loader.dataset)   # Calculate the average loss per sample
        epoch_accuracy = correct / total                        # Calculate the accuracy
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Evaluate on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        
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
    get_classification_report(model, test_loader)

# Evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()    # Set the model to evaluation mode
    test_loss = 0.0 # Initialize the test loss
    correct = 0     # Initialize the number of correct predictions
    total = 0       # Initialize the total number of predictions

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"): # Iterate over the test data
            inputs, labels = inputs.to(device), labels.to(device)   # Move inputs and labels to the device
            outputs = model(inputs)                                 # Forward pass
            loss = criterion(outputs, labels)                       # Calculate the loss
            test_loss += loss.item() * inputs.size(0)               # Accumulate the test loss
            predicted = torch.round(torch.sigmoid(outputs))         # Get the predicted labels
            total += labels.size(0)                                 # Increment the total count
            correct += (predicted == labels).sum().item()           # Increment the correct count

    accuracy = correct / total              # Calculate the accuracy
    test_loss /= len(test_loader.dataset)   # Calculate the average test loss per sample

    return test_loss, accuracy

# Function to get classification report
def get_classification_report(model, test_loader):
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

# -------------------------------------------------------------------------------------------------
# Define model parameters
lr = 0.001
batch_size = 64
dropout_rate = 0.5
input_dim = X_train.shape[1]
hidden_dim = 128
num_layers = 2  # Number of LSTM layers

# Create DataLoaders
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model, send it to the device
model = LSTMModel(input_dim, hidden_dim, num_layers, dropout_rate)
model.to(device)
print(model)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Initialize TensorBoard writer with unique name based on current date and time
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(f'runs/pytorch_lstm_experiment_{timestamp}')

# Log hyperparameters
hparams = {
    'lr': lr,
    'batch_size': batch_size,
    'hidden_dim': hidden_dim,
    'num_layers': num_layers,
    'dropout_rate': dropout_rate
}
writer.add_hparams(hparams, {})

# Train the model
train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, patience=10, epochs=100)

# Evaluate the model
evaluate(model, test_loader, criterion)

# Close the TensorBoard writer
writer.close()
