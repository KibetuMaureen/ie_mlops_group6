import pandas as pd
import yaml
from src.data_validation.validator import validate_schema, DataValidationError

# Loading the schema from the config
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Loading the real test data
df_test = pd.read_csv("../data/raw/fraudTest.csv")  # Using my existing path

# Running validation
try:
    validate_schema(df_test, config["data_validation"]["schema"]["columns"])
    print("Test data passed validation.")
except DataValidationError as e:
    print(f"Test data failed validation: {e}")
