"""
This function take the original files and extract the first 1000000 rows for each category
"""

import pandas as pd
from gobal_variables import *

max_size = 1000000
sample_size = 100000

# Creating an empty DataFrame with specified columns
df = pd.DataFrame(columns=["main_category", "title", "features"])

# Loop through each dataset and extract the first 1 million rows
for index in range(len(DATASET)):
    chunks = pd.read_json(path_or_buf=f"{RAW_DATA_FOLDER}/{DATASET[index]}.jsonl", lines=True, chunksize=sample_size)

    for id, c in enumerate(chunks):
        # Join into a paraphraph 
        c["features"] = c["features"].apply(lambda x: " ".join(x))
        # Set consistant label
        c["main_category"] = LABELS[index]
        # Join data snippet to df
        df = pd.concat([df, c[COLUMN_SELECTIONS]])
        # Stop at predefined number of iterations
        if id == int(max_size / sample_size) - 1:
            break

    # Store extracted data
    df.to_parquet(f"../../data/raw/{LABELS[index]}.parquet", compression="gzip")
    print(f"Succesfully extract {LABELS[index]}")