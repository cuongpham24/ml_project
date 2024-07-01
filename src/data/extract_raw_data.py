import pandas as pd

# Update this source folder appropriately 
RAW_DATA_FOLDER = "../../data/raw"

# Define feature columns 
dataset = ["meta_Beauty_and_Personal_Care", "meta_Books", "meta_Home_and_Kitchen"]
labels = ["personal_care", "book", "home"]
column_selections = ["main_category", "title", "features"]

max_size = 1000000
sample_size = 100000

# Creating an empty DataFrame with specified columns
df = pd.DataFrame(columns=["main_category", "title", "features"])

# Loop through each dataset and extract the first 1 million rows
for index in range(len(dataset)):
    chunks = pd.read_json(path_or_buf=f"{RAW_DATA_FOLDER}/{dataset[index]}.jsonl", lines=True, chunksize=sample_size)

    for id, c in enumerate(chunks):
        # Join into a paraphraph 
        c["features"] = c["features"].apply(lambda x: " ".join(x))
        # Set consistant label
        c["main_category"] = labels[index]
        # Join data snippet to df
        df = pd.concat([df, c[column_selections]])
        # Stop at predefined number of iterations
        if id == int(max_size / sample_size) - 1:
            break

    # Store extracted data
    df.to_parquet(f"../../data/raw/{labels[index]}.parquet", compression="gzip")
    print(f"Succesfully extract {labels[index]}")