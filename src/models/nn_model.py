import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
import numpy as np
from numpy import load

# Setup backend device
if torch.cuda.is_available():
    # Check for CUDA (traditional GPUs)
    device = torch.device("cuda")
    print("PyTorch is using CUDA.")
elif torch.backends.mps.is_available():
    # Check for MPS (Apple Silicon GPUs)
    device = torch.device("mps")
    print("PyTorch is using MPS.")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU.")

# Define the model
class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dense2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dense2(lstm_out)
        return out

class ClassificationModel_02(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_lstm_layers):
        super(ClassificationModel_02, self).__init__()
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_lstm_layers):
            self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))
            self.dropout_layers.append(nn.Dropout(0.3))
        
        self.dense2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)
        
        x = x[:, -1, :]  # Get the output of the last LSTM cell
        out = self.dense2(x)
        return out
    
def model_train_and_eval(model, train_loader, test_loader, learning_rate=0.001, num_epochs=40, device=device):
    
    # Initialize the model, loss function, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation loop
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return all_targets, all_predictions
    
def batch_data_loader(inputs, targets, train_mode=True, test_size=0.8, batch_size=32):
    # Convert to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    # Create dataset
    dataset = TensorDataset(inputs, targets)

    # Split dataset into training and testing sets
    train_size = int(test_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if train_mode:
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        # Create data loaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_mode)
        return data_loader

def load_emb_data(file_name):
    # load dict of arrays
    dict_data = load(f"../../data/processed/{file_name}.npz")
    # extract the first array
    return dict_data['arr_0']

if __name__ == "__main__":
    # Load data
    bert_vec = load_emb_data("feature_reduce_vec")
    index = load_emb_data("feature_reduce_index")
    label = load_emb_data("feature_reduce_label")

    train_loader, test_loader = batch_data_loader(bert_vec, label)

    # Hyperparameters
    input_size = 768
    hidden_size = 256
    num_classes = 3
    num_epochs = 75
    learning_rate = 0.0005
    num_lstm_layers = 4

    # Initialize model
    model = ClassificationModel_02(input_size, hidden_size, num_classes, num_lstm_layers)
    # Train and evaluate model
    all_targets, all_predictions = model_train_and_eval(model, train_loader, test_loader, learning_rate=learning_rate, num_epochs=num_epochs)