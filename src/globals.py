from cloudpathlib import CloudPath
from pathlib import Path

from src.versioning import get_poetry_version_no

# get current poetry version of this repo to tag tables and artifacts with
CURRENT_VERSION_NUM = get_poetry_version_no()

# local path for model artifacts
LOCAL_ARTIFACT_PATH = Path(__file__).parent.parent / "model_artifacts"

# gcs path for model artifacts
GCS_ARTIFACT_PATH = CloudPath("gs://the-cube/") / "export" / "surrogate_model" / "model_artifacts"
GCS_CURRENT_VERSION_ARTIFACT_PATH = GCS_ARTIFACT_PATH / CURRENT_VERSION_NUM 

# table paths (without version numbers)
CATALOG = 'ml'
SUMO_DB = 'surrogate_model'
MEGASTOCK_DB = 'megastock'

# Processed Data Table Paths
BUILDING_METADATA_TABLE = f'{CATALOG}.{SUMO_DB}.building_metadata'
MEGASTOCK_BUILDING_METADATA_TABLE = f'{CATALOG}.{MEGASTOCK_DB}.building_metadata'
ANNUAL_OUTPUTS_TABLE = f'{CATALOG}.{SUMO_DB}.building_simulation_outputs_annual'
WEATHER_DATA_TABLE = f'{CATALOG}.{SUMO_DB}.weather_data_hourly'

# Feature Table Paths
BUILDING_FEATURE_TABLE = f'{CATALOG}.{SUMO_DB}.building_features' #NOTE: contains rows for all upgrades 
MEGASTOCK_BUILDING_FEATURE_TABLE = f'{CATALOG}.{MEGASTOCK_DB}.building_features' #NOTE: containes rows for baseline only
WEATHER_FEATURE_TABLE = f'{CATALOG}.{SUMO_DB}.weather_features_hourly'