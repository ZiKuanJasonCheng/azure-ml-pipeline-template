#!/usr/bin/python3
import joblib
import logger
import numpy as np
import os
from azureml.core import Dataset
from azureml.core.run import Run
from argparse import ArgumentParser
from datetime import datetime 
from sklearn.model_selection import train_test_split
from utils.connection import Connection
from utils.label_encoding import frequency_encoding
from utils.model import xgboost_model

# Run this file with command: python -m pipeline.train --plant <plant_code>

# Read parameters
parser = ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument("--series_id", required=True)
parser.add_argument("--model_file", required=True)
parser.add_argument("--mode", required=True)
args = parser.parse_args()

# Connect to DB
conn = Connection(args.mode)

# Load series ID
with open(args.series_id + "/series_id.txt") as f:
    for i, line in enumerate(f):
        # We only fetch the first line
        series_id = int(line.split("\n")[0])
        break

# Only if series_id != -1 (There are new data to be analyzed) will we continue the remaining process
if series_id != -1:
    myLogger = logger.getLogger(__name__)

    # Connect to Workspace
    run = Run.get_context().parent
    ws = run.experiment.workspace

    # Connect to Azure Blob Storage
    def_blob_store = ws.get_default_datastore()

    myLogger.info(f"Start training model, current_time: {datetime.now()}")
    conn.update_ai_process(series_id, "Training model")

    # Read training data from registered data assets
    dataset = Dataset.get_by_name(workspace=ws, name=f"{args.plant}_training_data")
    train = dataset.to_pandas_dataframe()
    myLogger.info(f"Sample of training data: {len(train)}")
    myLogger.info("Start label encoding...")

    # Create directory to store the label encoding mappings
    os.makedirs("./label_encoding", exist_ok=True)

    # Apply frequency encoding to X and y
    lst_cols = ["colA", "colB", "colC", "colD", "colE", "colF", "colG", "sourcer_code"]
    train_encoded, encoding_mappings = frequency_encoding(train, columns=lst_cols)

    # Save and upload the encoding mappings to Azure Blob
    np.save(f"./label_encoding/{args.plant}_encoding_mappings.npy", encoding_mappings)
    def_blob_store.upload_files(files=[f"./label_encoding/{args.plant}_encoding_mappings.npy"], target_path="label_encoding", overwrite=True)
    myLogger.info("Label encoding done.")
        
    # Split to x and y
    X1 = train[["colA", "colB", "colC", "colD", "colE"]]  # Model 1
    X2 = train[["colC", "colD", "colE", "colF", "colG"]]  # Model 2
    y = train["sourcer_code"]

    # Split into training and validation set
    x_train1, x_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.1, random_state=597)
    x_train2, x_val2, y_train2, y_val2= train_test_split(X2, y, test_size=0.1, random_state=597)

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.05, 0.01, 0.001],
        'n_estimators': [25, 50, 100, 200],
    }

    # Create a directory to store models
    os.makedirs(args.model_file, exist_ok=True)

    # Create XGBoost model 1
    myLogger.info("Training model with component types...")
    xgboost_model1, accuracy1 = xgboost_model(x_train1, y_train1, x_val1, y_val1)
    myLogger.info(f"Model with component types training has been done, current_time: {datetime.now()}")

    joblib.dump(xgboost_model1, f"{args.model_file}/{args.plant}_model.joblib")
    myLogger.info("Model with component types had been saved!")

    # Create XGBoost model 2
    myLogger.info("Training model without component types...")
    xgboost_model2, accuracy2 = xgboost_model(x_train2, y_train2, x_val2, y_val2)
    myLogger.info(f"Model without component types training has been done, current_time: {datetime.now()}")

    joblib.dump(xgboost_model2, f"{args.model_file}/{args.plant}_model_no_comp.joblib")
    myLogger.info("Model without component types had been saved!")

    conn.update_ai_process(series_id, "Training model done")