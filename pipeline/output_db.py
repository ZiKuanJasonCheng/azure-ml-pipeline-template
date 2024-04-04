#!/usr/bin/python3
import logger
import psycopg2
import psycopg2.extras as extras
from argparse import ArgumentParser
from azureml.core import Dataset
from azureml.core.run import Run
from configparser import ConfigParser
from datetime import datetime
from utils.connection import Connection

# Run this file with command: python -m pipeline.upload_db --plant <plant_code>

myLogger = logger.getLogger(__name__)

cur_time = datetime.now()
myLogger.info(f"Start uploading data to DB, current_time: {cur_time}")

# Connect to Workspace
run = Run.get_context().parent
ws = run.experiment.workspace

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
    # Read config
    config = ConfigParser()
    config.read('config.ini')

    conn.update_ai_process(series_id, "Upload to DB")

    # Read result
    myLogger.info("Reading predicting result...")
    dataset = Dataset.get_by_name(workspace=ws, name=f"{args.plant}_result")
    pred_result = dataset.to_pandas_dataframe()

    # Upload data to DB
    # ...

    conn.update_ai_process(series_id, "Y")
    conn.update_analysis_console(series_id, "data_ready", "Y")
    conn.update_analysis_console(series_id, "ai_process_end_datetime", datetime.now())
    myLogger.info("DB connecting closed...")