import os
from pathlib import Path
import pyspark.sql.functions as F
import toml
import re

if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from databricks.sdk.runtime import *


def get_poetry_version_no():
    """Get current version of this repo with zero-padded version numbers."""
    repo_dir = Path(__file__).parent.parent
    project_data = toml.load(repo_dir / "pyproject.toml")
    poetry_version_no = project_data["tool"]["poetry"]["version"]

    # Zero-pad each component of the version
    if re.fullmatch(r'\d+\.\d+\.\d+', poetry_version_no):
        return  "_".join(f"{int(part):02}" for part in poetry_version_no.split("."))
    else:
        return poetry_version_no
    

def get_most_recent_table_version(full_table_name, max_version='current_version', return_version_number_only=False):
    """
    Get the most recent version of a table defined as the table with the highest zero-padded, underscore-delimited
    semantic version suffix.

    If max_version is passed, then find the most recent version that is no higher than this.

    Parameters:
        full_table_name (str): Table name in the format `{catalog}.{database}.{table}`
        max_version (str, optional): Max version to cap the table versions at (e.g., '02_05_10'). Defaults to the current version number.
        return_version_number_only (bool, optional): Whether to return just the version number string. Defaults to False.

    Returns:
        str or None: Either the full table name or just the version number (if `return_version_number_only=True`).
    """
    try:
        catalog, database, table = full_table_name.split(".")
    except ValueError:
        raise ValueError("Invalid table name format: must be in the form 'catalog.database.table'")

    if max_version == 'current_version':
        max_version = get_poetry_version_no()
        # if doesnt match a semantic version number, just ignore since ordinality does
        if not re.fullmatch(r'\d+\.\d+\.\d+', max_version):
            max_version=None

    # List all tables in the catalog and database
    tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{database}")

    # Filter tables that match the table name with a semantic version pattern
    filtered_tables = tables_df.filter(F.col("tableName").rlike(rf"^{table}_\d+_\d+_\d+$"))

    # If max_version is provided, filter tables to be less than or equal to max_version
    if max_version:
        formatted_max_version = "_".join(f"{int(part):02}" for part in max_version.split("_"))
        filtered_tables = filtered_tables.filter(F.col("tableName") <= f"{table}_{formatted_max_version}")

    # Order the filtered tables by version in descending order (most recent first)
    table_names = (
        (filtered_tables.orderBy(F.desc(F.col("tableName"))).select("tableName")).rdd.flatMap(lambda x: x).collect()
    )

    # If no matching tables are found, return None
    if not table_names:
        return None

    # Extract just the version number if requested
    if return_version_number_only:
        version_number = table_names[0].replace(f"{table}_", "")
        return version_number

    return f"{catalog}.{database}.{table_names[0]}"
