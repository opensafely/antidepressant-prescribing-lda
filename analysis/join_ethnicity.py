######################################

# This script adds the ethnicity variable to the main input data

######################################

# --- IMPORT STATEMENTS ---

## Import packages
import pandas as pd
import os

## Import data
# ethnicity_df = pd.read_feather('output/data/input_ethnicity.feather')
ethnicity_df = pd.read_csv("output/input_ethnicity.csv")

# --- ADD ETHNICITY ---

for file in os.listdir("output/"):
    if file.startswith("input"):
        # exclude ethnicity
        if file.split("_")[1] not in ["ethnicity.csv", "practice", "flow"]:
            # file_path = os.path.join('output/data', file)
            file_path = os.path.join("output", file)
            # df = pd.read_feather(file_path)
            df = pd.read_csv(file_path)
            merged_df = df.merge(ethnicity_df, how="left", on="patient_id")

            # merged_df.to_feather(file_path)
            merged_df.to_csv(file_path, index=False)
