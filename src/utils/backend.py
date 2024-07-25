import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import numpy as np

# Setup backend device
def set_device():
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
    return device


def model_train_and_eval(model, train_loader, test_loader, learning_rate=0.001, num_epochs=40, device="cpu"):
    
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
    dict_data = np.load(f"data/processed/{file_name}.npz")
    # extract the first array
    return dict_data['arr_0']
