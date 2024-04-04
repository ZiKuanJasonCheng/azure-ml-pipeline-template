#!/usr/bin/python3
import json
import os
import pandas as pd
import logger
from argparse import ArgumentParser
from azureml.core import Dataset
from azureml.core.run import Run
from datetime import datetime
from datetime import timedelta
from utils.connection import Connection

# Run this file with command: python -m pipeline.read_data --plant <plant_code>

# Read parameters
parser = ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument("--series_id", required=True)
parser.add_argument('--mode', required=True)
args = parser.parse_args()

# Logger
myLogger = logger.getLogger(__name__)

# Connect to Workspace
run = Run.get_context().parent
ws = run.experiment.workspace

# Connect to Azure Blob Storage
def_blob_store = ws.get_default_datastore()

# Connect to DB
conn = Connection(args.mode)

# Find unfinished jobs in analysis_console by plant
myLogger.info("Check analysis console status...")
myLogger.info(f"plant: {args.plant}")
myLogger.info(f"mode: {args.mode}")
control_table = conn.fetch_control_table(args.plant)
myLogger.info(f"control_table: {control_table.head()}")

if control_table is not None:  # If no error occurred while fetching control_table, then it should not be None
    if control_table.empty:
        myLogger.info("There is no new data.")

        # Save series_id
        os.makedirs(args.series_id, exist_ok=True)  # Create a folder to save id

        with open(args.series_id + "/series_id.txt", 'w') as f:
            f.write("-1")

    else:  # If there exists new data
        try:
            myLogger.info(f"control table: \n{control_table.head()}")
            series_id = control_table.iloc[0]['id']
            plant = control_table.iloc[0]['plant']
            mrp_run_date = control_table.iloc[0]['mrp_run_date']
            batch_id = control_table.iloc[0]['batch_id']
            post_datetime = control_table.iloc[0]['post_datetime']

            # Save series_id
            os.makedirs(args.series_id, exist_ok=True) # Create folders to save id
            with open(args.series_id + "/series_id.txt", 'w') as file:
                file.write(str(series_id))

            # Read data from mrp_sourcer_code_analysis
            myLogger.info(f'id: {series_id}, plant: {plant}, mrp_run_date : {mrp_run_date}, batch_id: {batch_id}, post_datetime : {post_datetime}')
            mrp_sourcer_code = conn.fetch_mrp_sourcer_code_table(plant, mrp_run_date, batch_id, post_datetime)

            # Check if length of control table recorded data and length of real data are different
            if len(mrp_sourcer_code) != control_table.iloc[0]['total_record']:
                myLogger.info(f"mrp_sourcer_code: {len(mrp_sourcer_code)}")
                myLogger.info(f"control_table: {control_table.iloc[0]['total_record']}")
                myLogger.exception("Data lengths are different")
                conn.update_ai_process(series_id, "Data lengths are different")
            else:
                myLogger.info(f"Start predicting sourcer code, ID: {series_id}")
                cur_time = datetime.now()
                myLogger.info(f"Start prepare_data, current_time: {cur_time}")
                conn.update_ai_process(series_id, "Preparing data")

                # Fetch date within 1 day
                date = mrp_run_date - timedelta(days=1)
                date = date.strftime('%Y%m%d') + "000000000"
                date = datetime.strptime(date, '%Y%m%d%H%M%S%f').strftime('%Y-%m-%d %H:%M:%S')

                # Data preprocessing step...
                # ...
                try:
                    conn.update_analysis_console(series_id, "ai_process_start_datetime", datetime.now())
                    
                    # ...
                    mrp_preprocessed = ...
                    mrp_preprocessed = mrp_preprocessed.reset_index()

                    myLogger.info(mrp_preprocessed.head())

                    # Export preprocessed data
                    path = f"./data/preprocessed/{plant}_preprocessed.csv"
                    mrp_preprocessed.to_csv(path, index=False)

                    # Upload to blob storage
                    def_blob_store.upload_files(files=[path], target_path="preprocessed", overwrite=True)

                    # Register dataset
                    tabular = Dataset.Tabular.from_delimited_files(def_blob_store.path(f"./preprocessed/{plant}_preprocessed.csv"))
                    df_register = tabular.register(workspace=ws, name=f"{plant}_preprocessed_data", create_new_version=True)
                    
                    # Split into training and inference data (sourcer code is not NULL)
                    train = mrp_preprocessed.loc[mrp_preprocessed["sourcer_code"] != '', :]
                    inference = mrp_preprocessed.loc[mrp_preprocessed["sourcer_code"] == '', :]

                    # Keep the first sourcer code only
                    train.loc[:, "sourcer_code"] = train["sourcer_code"].apply(lambda x: x.split(";")[0])

                    # Export training and inference dataset
                    path = f"./data/train/{plant}_train.csv"
                    train.to_csv(path, index=False)

                    # Upload to blob storage
                    def_blob_store.upload_files(files=[path], target_path="train", overwrite=True)

                    # Register dataset
                    tabular = Dataset.Tabular.from_delimited_files(def_blob_store.path(f"./train/{plant}_train.csv")) # Different with local path
                    df_register = tabular.register(workspace=ws, name=f"{plant}_training_data", create_new_version=True)

                    path = f"./data/inference/{plant}_inference.csv"
                    inference.to_csv(path, index=False)

                    # Upload to blob storage
                    def_blob_store.upload_files(files=[path], target_path="inference", overwrite=True)

                    # Register dataset
                    tabular = Dataset.Tabular.from_delimited_files(def_blob_store.path(f"./inference/{plant}_inference.csv")) # Different with local path
                    df_register = tabular.register(workspace=ws, name=f"{plant}_inference_data", create_new_version=True)

                    # If no inference data exists
                    if inference.empty:
                        myLogger.info("There is no inference data.")
                        conn.update_ai_process(series_id, "No inference data")
                except:
                    myLogger.exception("An error occurred while proceeding mrp_preprocessed")

                conn.update_ai_process(series_id, "Preparing data done")
                myLogger.info(f"prepare_data DONE. ID: {series_id}")
                myLogger.info(f"prepare_data DONE. Time consumption: {(datetime.now() - cur_time).total_seconds()} sec")
        except:
            myLogger.exception(f"An error occurred while analyzing {plant}")