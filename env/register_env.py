from azureml.core import Environment
from ..utils.features import get_workspace

# Run this file with command: python -m env.register_env

ws = get_workspace()
print("Found workspace {} at location {}".format(ws.name, ws.location))

# Register enviroment from Docker image
env = Environment.from_conda_specification(name='my-custom-env', file_path='./env/conda_env.yml')
env = env.register(workspace=ws)
env.build(workspace=ws)