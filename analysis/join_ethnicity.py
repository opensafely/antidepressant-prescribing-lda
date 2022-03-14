######################################

# This script adds the ethnicity variable to the main input data

######################################

# --- IMPORT STATEMENTS ---

## Import packages
import pandas as pd
import pathlib

## Import data
# ethnicity_df = pd.read_feather('output/data/input_ethnicity.feather')
ethnicity_df = pd.read_csv("output/input_ethnicity.csv")

# --- ADD ETHNICITY ---
for path in pathlib.Path("output").rglob("input*"):
    if "ethnicity" not in path.name:
        # df = pd.read_feather(path.name)
        df = pd.read_csv(path)
        merged_df = df.merge(ethnicity_df, how="left", on="patient_id")

        # merged_df.to_feather(path.name, index=False)
        merged_df.to_csv(path, index=False)
