from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
import numpy as np
import pandas as pd
from math import ceil
from numpy import load

# Hyperparameters
input_size = 768
hidden_size = 128
num_classes = 3
num_epochs = 20
learning_rate = 0.001

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

