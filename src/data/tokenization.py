import os
import sys
from math import ceil
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
from src.data.constants import *

def get_raw_data(label, column):
    # Validate inputs
    assert label in LABELS, "Incorect file's name. Valid labels are book, home, personal_care"
    assert column in ["title", "features"], "Incorect column's name. Valid labels are title, features"

    # Read the extracted raw data
    df_temp = pd.read_parquet(f"../../data/raw/{label}.parquet")
    df_temp[column] = df_temp[column].astype("str")
    # Convert text to lower case
    df_temp = df_temp.drop("main_category", axis=1).apply(lambda x: x.str.lower())
    # Convert non-empty values in the input column to list
    documents = df_temp[column][df_temp[column] != ""].to_list()
    
    return documents

# Function to get embeddings for a batch of texts
def word_to_vec(texts_batch):
    inputs = tokenizer(texts_batch, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()

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

if __name__ == "main":
    # Get parameters
    label = sys.argv[1]    # The first argument
    column = sys.argv[2]   # The second argument
    documents = get_raw_data(label, column)
    device = set_device()
    
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Send data to GPU
    model.to(device)

    # Tokenize and vectorize data
    num_batch = ceil(len(documents) / batch_size)
    for i in range(num_batch):
        if i % 4 == 0:
            print(f"Batch {i+1}") 
        if i == 0:
            xtrain = word_to_vec(documents[i * batch_size:(i + 1) * batch_size])
        elif i == num_batch - 1:
            xtrain = np.vstack([xtrain, word_to_vec(documents[i * batch_size:len(documents)])])
        else:
            xtrain = np.vstack([xtrain, word_to_vec(documents[i * batch_size:(i + 1) * batch_size])])

    # save numpy array as npz file
    np.savez_compressed(f"../../data/processed/{label}_bert_{column}.npz", all_embeddings_np)
