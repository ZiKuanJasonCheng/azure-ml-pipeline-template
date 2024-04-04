import logger
from argparse import ArgumentParser
from azureml.core import Dataset
from azureml.core import Model
from azureml.core.run import Run

# Run this file with command: python -m pipeline.register_model --plant <plant_code>

myLogger = logger.getLogger(__name__)

# Read parameters
parser = ArgumentParser()
parser.add_argument('--plant', required=True)
parser.add_argument("--model_file", required=True)
parser.add_argument("--series_id", required=True)
args = parser.parse_args()

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

      # Model names
      model_name = f"{args.plant}_model.joblib"
      model_name_no_comp = f"{args.plant}_model_no_comp.joblib"

      # Read training data
      dataset = Dataset.get_by_name(ws, f"{args.plant}_training_data", version="latest")

      # Register model 1
      model = Model.register(workspace=ws,
                             model_path=args.model_file,
                             model_name= model_name,
                             tags={
                                   "tag1": "mytag1",
                                   "tag2": "mytag2",
                                   "pipeline_id": run.id,
                                   "dataset": f"{args.plant}_training_data: {dataset.version}"
                                  },
                             properties={
                                         "accuracy_bottomline": 0.8,
                                         "accuracy_target": 0.8
                                        },
                             description="My description for model 1")

      myLogger.info(f"Name of Model 1: {model.name}")
      myLogger.info(f"Version of Model 1: {model.version}")

      # Register model 2
      model_2 = Model.register(workspace=ws,
                               model_path=args.model_file,
                               model_name= model_name_no_comp,
                               tags={
                                     "tag1": "mytag1",
                                     "tag2": "mytag2",
                                     "pipeline_id": run.id,
                                     "dataset": f"{args.plant}_training_data: {dataset.version}"
                                    },
                               properties={
                                           "accuracy_bottomline": 0.8,
                                           "accuracy_target": 0.8
                                          },
                               description="My description for model 2")

      myLogger.info(f"Name of Model 2: {model_2.name}")
      myLogger.info(f"Version of Model 2: {model_2.version}")