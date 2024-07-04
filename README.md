# 1. Environment setup
Create a conda virtual environment with
```
conda create --name <env> --file requirements.txt
```
# 2. Extract data from souce
Execute `extract_raw_data.py` in `src/data` to extract raw data from the original data source
- Set the correct directory for `RAW_DATA_FOLDER`
- Extracted data will be in `./data/raw` 