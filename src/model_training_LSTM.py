import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import datetime
from src.ml_utils_LSTM import SequenceDataset, ConfigurableLSTM, train, evaluate

# Load the data splits
X_train = joblib.load('data/processed/v2/splits/base/X_train.pkl')
X_test = joblib.load('data/processed/v2/splits/base/X_test.pkl')
y_train = joblib.load('data/processed/v2/splits/base/y_train.pkl')
y_test = joblib.load('data/processed/v2/splits/base/y_test.pkl')

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
model = ConfigurableLSTM(input_dim, hidden_dim, num_layers, dropout_rate)
model.to(device)
print(model)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Initialize TensorBoard writer with unique name based on current date and time
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(f'runs/pytorch_lstm_experiment_{timestamp}_TEST')

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
train(model, train_loader, test_loader, criterion, optimizer, scheduler, writer, device, patience=10, epochs=100)

# Close the TensorBoard writer
writer.close()
