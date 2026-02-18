import os

LAKE_CONFIG = {
        'subscription_id' : os.environ.get("SUBSCRIPTION_ID"),
        'resource_group' : os.environ.get("RESOURCE_GROUP"),
        'workspace_name' : os.environ.get("WORKSPACE_NAME"),
        'storage_account_name' : os.environ.get("STORAGE_ACCOUNT_NAME")
    }