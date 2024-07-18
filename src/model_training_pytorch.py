import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

# Custom Dataset class to handle sparse matrices
class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_data = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()
        y_data = torch.tensor(self.y.iloc[idx], dtype=torch.long)  # Using iloc for safe indexing
        return X_data, y_data

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

# Create DataLoader
train_dataset = SparseDataset(X_train, y_train)
test_dataset = SparseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the neural network
class ConfigurableNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.5):
        super(ConfigurableNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Example usage
input_dim = X_train.shape[1]
output_dim = len(y_train.unique())
hidden_dims = [512, 256, 128, 64]  # Number of units in each hidden layer

model = ConfigurableNN(input_dim, output_dim, hidden_dims)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function with early stopping
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, patience=5, epochs=10):
    best_loss = float('inf')
    current_patience = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        
        # Evaluate on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Check for early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
    # After training, get classification report
    get_classification_report(model, test_loader)

# Evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    test_loss /= len(test_loader.dataset)
    
    return test_loss, accuracy

# Function to get classification report
def get_classification_report(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Train the model
train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100)


# Evaluate the model
evaluate(model, test_loader, criterion)

# Save the model
torch.save(model.state_dict(), 'models/v2/base/pytorch_nn_model.pth')
print('Saved the PyTorch model trained on the full dataset.')
