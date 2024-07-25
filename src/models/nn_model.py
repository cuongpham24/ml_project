import sys
import os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
import numpy as np
from src.utils.backend import set_device, model_train_and_eval, batch_data_loader, load_emb_data

# Setup backend device
device = set_device()

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
    
if __name__ == "__main__":
    # Load data
    bert_vec = load_emb_data("feature_reduce_vec")
    index = load_emb_data("feature_reduce_index")
    label = load_emb_data("feature_reduce_label")

    train_loader, test_loader = batch_data_loader(bert_vec, label)

    # Hyperparameters
    input_size = 768
    hidden_size = 64
    num_classes = 3
    num_epochs = 1
    learning_rate = 0.0005
    num_lstm_layers = 4

    # Initialize model
    model = ClassificationModel_02(input_size, hidden_size, num_classes, num_lstm_layers)
    # Train and evaluate model
    all_targets, all_predictions = model_train_and_eval(model, train_loader, test_loader, learning_rate=learning_rate, num_epochs=num_epochs)