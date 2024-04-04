#!/usr/bin/python3
import argparse
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import StepSequence
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from utils.features import get_workspace

# Run this file with command: python -m azure_ml_pipeline --plant <plant_code>

ws = get_workspace()
print("Found workspace {} at location {}".format(ws.name, ws.location))

# Get resourecs
def_blob_store = ws.get_default_datastore()  # To store temporary input/output data during pipeline execution
aml_compute = AmlCompute(workspace=ws, name="vm-compute")  # Compute instance for data preprocessing, training models, etc. (Check compute name: Manage - Compute - Compute clusters)
env = Environment.get(workspace=ws, name="my-custom-env")  # Environment for Docker settings, packages used by pipeline, and environment variables (Check environment name: Assets - Environments)

aml_run_config = RunConfiguration()
aml_run_config.target = aml_compute
aml_run_config.environment = env

# Read parameters from Gitlab-CI yaml
parser = argparse.ArgumentParser()
parser.add_argument("--plant", required=True)
parser.add_argument("--mode", required=True)
args = parser.parse_args()

# Define pipline parameters
plant = PipelineParameter(name="plant", default_value=args.plant)
mode = PipelineParameter(name="mode", default_value=args.mode)

# Define pipeline data
# PipelineData serves as a pipeline connection between steps. It can be used as a medium for passing intermediate input/output data.
series_id = PipelineData("series_id", datastore=def_blob_store)
model = PipelineData("model", datastore=def_blob_store)
encoding_mappings = PipelineData("encoding_mappings", datastore=def_blob_store)

# Setup pipeline steps
source_directory = "./"

step_read_data = PythonScriptStep(name="read_data",
                                    script_name="pipeline/read_data.py",
                                    arguments=[
                                        "--series_id", series_id,
                                        "--plant", plant,
                                        "--mode", mode,
                                    ],
                                    inputs=[],
                                    outputs=[
                                        series_id
                                    ],
                                    runconfig=aml_run_config,
                                    source_directory=source_directory,
                                    allow_reuse=False)

step_train = PythonScriptStep(name="train",
                              script_name="pipeline/train.py",
                              arguments=[
                                  "--series_id", series_id,
                                  "--plant", plant,
                                  "--model_file", model,
                                  "--mode", mode,
                              ],
                              inputs=[
                                  series_id,
                              ],
                              outputs=[
                                  encoding_mappings,
                                  model,
                              ],
                              runconfig=aml_run_config,
                              source_directory=source_directory,
                              allow_reuse=False)

step_register_model = PythonScriptStep(name="register_model",
                                 script_name="pipeline/register_model.py",
                                 arguments=[
                                     "--series_id", series_id,
                                     "--model_file", model,
                                     "--plant", plant,
                                 ],
                                 inputs=[
                                     series_id,
                                     encoding_mappings,
                                     model
                                 ],
                                 outputs=[
                                 ],
                                 runconfig=aml_run_config,
                                 source_directory=source_directory,
                                 allow_reuse=False)
                                 
step_predict = PythonScriptStep(name="predict",
                                script_name="pipeline/predict.py",
                                arguments=[
                                    "--series_id", series_id,
                                    "--plant", plant,
                                    "--mode", mode,
                                ],
                                inputs=[
                                    series_id,
                                    encoding_mappings,
                                    model
                                ],
                                outputs=[
                                ],
                                runconfig=aml_run_config,
                                source_directory=source_directory,
                                allow_reuse=False)

step_upload_db = PythonScriptStep(name="upload_db",
                                  script_name="pipeline/upload_db.py",
                                  arguments=[
                                      "--series_id", series_id,
                                      "--plant", plant,
                                      "--mode", mode,
                                  ],
                                  inputs=[
                                      series_id,
                                  ],
                                  outputs=[
                                  ],
                                  runconfig=aml_run_config,
                                  source_directory=source_directory,
                                  allow_reuse=False)

step_sequence = StepSequence(steps=[step_read_data, step_train, step_register_model, step_predict, step_upload_db])
pipeline = Pipeline(workspace=ws, steps=step_sequence)
#print(f"pipeline: {pipeline}")

# Submit pipeline experiment
pipeline_run = Experiment(workspace=ws, name='my_ml_pipeline').submit(config=pipeline, regenerate_outputs=False, tags={"plant": args.plant})
#print(f"pipeline_run: {pipeline_run}")