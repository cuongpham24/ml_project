# Update this source folder appropriately 
RAW_DATA_FOLDER = "../../data/raw"

# Define feature columns 
DATASET = ["meta_Beauty_and_Personal_Care", "meta_Books", "meta_Home_and_Kitchen"]
LABELS = ["personal_care", "book", "home"]
COLUMN_SELECTIONS = ["main_category", "title", "features"]
LABEL_TO_ID = {"personal_care": 0, "book": 1, "home": 2}