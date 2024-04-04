#!/usr/bin/python3
import joblib
import logger
import numpy as np
import os
import pandas as pd
from azureml.core import Dataset
from azureml.core.model import Model
from azureml.core.run import Run
from argparse import ArgumentParser
from utils.connection import Connection

# Run this file with command: python -m pipeline.predict --plant <plant_code>

myLogger = logger.getLogger(__name__)

# Read parameters
parser = ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument("--series_id", required=True)
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
    # Connect to Workspace
    run = Run.get_context().parent
    ws = run.experiment.workspace

    # Connect to Azure Blob Storage
    def_blob_store = ws.get_default_datastore()

    conn.update_ai_process(series_id, "Predicting inference")

    # Read inference data
    myLogger.info("Reading inference data...")
    dataset = Dataset.get_by_name(workspace=ws, name=f"{args.plant}_inference_data")
    inference_raw = dataset.to_pandas_dataframe()
    inference = inference_raw.copy()

    # Read label encoding mappings
    def_blob_store.download(target_path=f'.', prefix=f"label_encoding/{args.plant}_encoding_mappings.npy", overwrite=True)
    encoding_mappings = np.load(f"label_encoding/{args.plant}_encoding_mappings.npy", allow_pickle=True).item()

    # Apply label encoding using the loaded mappings, if not mapped then apply -1
    myLogger.info("Label encoding inference data...")
    for column in encoding_mappings.keys():
        inference[column] = inference[column].apply(lambda x: encoding_mappings[column].get(x, -1))

    # Split inference if columns colA, colB are NAs
    inference_1 = inference[(inference["colA"]!=-1) & (inference["colB"]!=-1)]
    inference_2 = inference[(inference["colA"]==-1) & (inference["colB"]==-1)]

    # Create empty dataframe
    result_1 = pd.DataFrame()
    result_2 = pd.DataFrame()

    # Predict: case 1 - with colA, colB
    if not inference_1.empty:
        myLogger.info("Loading model with component types...")
        path = Model.get_model_path(f"{args.plant}_model.joblib", _workspace=ws) 
        model = joblib.load(path+f"/{args.plant}_model.joblib")

        myLogger.info("Predicting case 1 - with colA, colB...")
        inference_1 = inference_1[["colA", "colB", "colC", "colD", "colE"]]  # Features of model 1

        # Top 3 predictions
        pred_y = model.predict_proba(inference_1)
        pred_top3 = pd.DataFrame(np.argsort(pred_y, axis=1)[:, -3:][:, ::-1], columns=[f"prediction{i+1}" for i in range(3)])
        result = inference_raw.iloc[inference_1.index, :].copy().reset_index(drop=True)
        print(result.head())

        result_1 = pd.concat([result, pred_top3], axis=1)

    # Predict: case 2 - no colA, colB
    if not inference_2.empty:
        myLogger.info("Loading model without component types...")
        path = Model.get_model_path(f"{args.plant}_model_no_comp.joblib", _workspace=ws) 
        model_no_comp = joblib.load(path+f"/{args.plant}_model_no_comp.joblib")

        myLogger.info("Predicting case 2 - no colA, colB...")
        inference_2 = inference_2[["colC", "colD", "colE", "colF", "colG"]]  # Features of model 2

        # Top 3 predictions
        pred_y = model_no_comp.predict_proba(inference_2)
        pred_top3 = pd.DataFrame(np.argsort(pred_y, axis=1)[:, -3:][:, ::-1], columns=[f"prediction{i+1}" for i in range(3)])
        result = inference_raw.iloc[inference_2.index, :].copy().reset_index(drop=True)
        print(result.head())

        result_2 = pd.concat([result, pred_top3], axis=1)

    # Combine case 1 and case 2
    myLogger.info("Final result:")
    final_result = pd.concat([result_1, result_2], axis=0)

    # Replace sourcer_code with the original value using encoding_mappings, from values to keys
    reverse_mapping = {v: k for k, v in encoding_mappings["sourcer_code"].items()}  # Reverse dictionary

    for col in ["prediction1", "prediction2", "prediction3"]:
        final_result[col] = final_result[col].replace(reverse_mapping)
    myLogger.info(final_result.head())

    # Save to data/result
    os.makedirs("./data/result", exist_ok=True)
    path = f"./data/result/{args.plant}_result.csv"
    final_result.to_csv(path, index=False)

    # Upload to blob storage
    def_blob_store.upload_files(files=[path], target_path="result", overwrite=True)

    # Register dataset
    tabular = Dataset.Tabular.from_delimited_files(def_blob_store.path(f"./result/{args.plant}_result.csv"))  # Different with local path
    df_register = tabular.register(workspace=ws, name=f"{args.plant}_result", create_new_version=True)

    myLogger.info("Predicted data have been stored.")
    conn.update_ai_process(series_id, "Predicting inference done")