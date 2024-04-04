from configparser import ConfigParser
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

# General functions for Azure pipeline

# Read config
config = ConfigParser()
config.read('config.ini')

def get_workspace():
    # Connect to Workspace
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config.get("workspace", "tenant_id"),
        service_principal_id=config.get("workspace", "client_id"),
        service_principal_password=config.get("workspace", "client_secret")
    )

    ws = Workspace(
        subscription_id=config.get("azure", "subscription_id"),
        resource_group=config.get("azure", "resource_group"),
        workspace_name=config.get("azure", "workspace_name"),
        auth=svc_pr
    )

    return ws