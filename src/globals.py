from cloudpathlib import CloudPath
from pathlib import Path

from src.versioning import get_poetry_version_no

# get current poetry version of this repo to tag tables and artifacts with
CURRENT_VERSION_NUM = get_poetry_version_no()

# local path for model artifacts
LOCAL_ARTIFACT_PATH = Path(__file__).parent.parent / 'model_artifacts'

# gcs path for model artifacts
GCS_ARTIFACT_PATH = CloudPath("gs://the-cube/") / "export" / "surrogate_model" / "model_artifacts"