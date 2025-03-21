"""Functions for processing BSB (BuildStockBatch) outputs"""
# TODO: Move this into dmutils

import sys
import os
from cloudpathlib import CloudPath
import collections
from functools import reduce, partial
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType, DoubleType, LongType, StructField, StructType

from dmlutils import constants

sys.path.append("../src")

from utils import data_cleaning

if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from databricks.sdk.runtime import *


def get_schema_diffs(schemas: list[StructType]) -> StructType:
    """
    Given a list of schemas:
      - Finds fields that exist in multiple schemas with different types.
      - When possible, decides what type that field should be, and adds it to the schema that is returned.
      - Raises an exception if a type conflict can't be resolved.

    Args:
        schemas: List of schemas to resolve.

    Returns:
        Schema with only the fields that differ among the provided schemas,
        with the types we should use.
    """
    # Collect all the unique field definitions in a set
    all_fields = set()
    diff_fields = []
    for s in schemas:
        all_fields.update(s.fields)

    # Group by name to find conflicts
    fields_by_name = collections.defaultdict(list)
    for field in all_fields:
        fields_by_name[field.name].append(field)

    err_count = 0
    for f_name, fields in fields_by_name.items():
        if len(fields) == 1:
            continue

        type_set = {f.dataType for f in fields}
        # Cast long ints to doubles
        if type_set == {LongType(), DoubleType()}:
            diff_fields.append(StructField(f_name, DoubleType()))
        # Cast ints to booleans
        elif type_set == {LongType(), BooleanType()}:
            diff_fields.append(StructField(f_name, DoubleType()))
        else:
            err_count += 1
            print(f"{f_name} has multiple types that can't be resolved:")
            print(f"  {type_set}")

    if err_count:
        raise ValueError(f"Could not resolve {err_count} field(s)")

    # Return just the fields that might have conflicts
    return StructType(diff_fields)


def load_clean_and_add_upgrade_id(
    file_path: str,
    upgrade_id: float,
    schema: StructType = None,
    schema_diffs: StructType = None,
):
    """
    Load a DataFrame from a Parquet file, clean its column names, and add an upgrade_id column.

    Args:
    - file_path (str): Path to the Parquet file
    - upgrade_id (double): The value for the 'upgrade_id' column
    - schema (StructType, optional): Optional schema to enforce on the DataFrame
    - schema_diffs (StructType, optional): Schema containing fields that should be cast to other
        types if they don't already match.

    Returns:
    - DataFrame with cleaned column names and upgrade_id column if provided
    """

    if schema is not None:
        df = spark.read.schema(schema).parquet(str(file_path))
    else:
        df = spark.read.parquet(str(file_path))

    def get_dtype(df, colname):
        """Get the type of a column within a dataframe"""
        return [dtype for name, dtype in df.dtypes if name == colname][0]

    for field in schema_diffs.fields:
        if field.name in df.schema.fieldNames():
            if field.dataType.typeName() != get_dtype(df, field.name):
                # Escape column names with periods, because PySpark doesn't like them.
                # These are cleaned up later.
                colname = f"`{field.name}`" if "." in field.name else field.name
                df = df.withColumn(field.name, df[colname].cast(field.dataType))

    if "completed_status" in df.columns:
        df = df.where(F.col("completed_status") == "Success")

    df = df.withColumn("upgrade_id", F.lit(upgrade_id))

    return df


def read_combine_sims():
    """
    Read in RAStock sims and combine into a single Spark DF

    This function is getting revamped in a different repo so not bothering to document for now
    """
    # TODO: update with new code from natalie/mohammad in pep

    BSB_DATA_FPATH = CloudPath("gs://the-cube") / "data" / "raw" / "bsb_sims"
    upgrade_to_fpaths = {
        11.01: BSB_DATA_FPATH / "med_hp" / "max_load_sizing",
        11.02: BSB_DATA_FPATH / "med_hp" / "hers_sizing",
        11.03: BSB_DATA_FPATH / "med_hp" / "acca_maxload_sizing",
        11.04: BSB_DATA_FPATH / "med_hp_no_setback" / "MAX" / "all parquets",
        11.05: BSB_DATA_FPATH / "med_hp_no_setback" / "HERS" / "all parquets",
        11.06: BSB_DATA_FPATH / "med_hp_no_setback" / "ACCA_MAX" / "all parquets",
        13.01: BSB_DATA_FPATH / "med_hp_weatherization" / "no_setpoint_setback" / "all parquets",
        14.01: BSB_DATA_FPATH / "gshp_full_data" / "ground_source_heatpump_HERS_maxload" / "all parquets",
        14.02: BSB_DATA_FPATH / "gshp_full_data" / "hers_no_setpoint_setback" / "all parquets",
    }

    # Get the schema for each individual file
    initial_dataframes = [
        spark.read.parquet(str(fpath.path))
        for fpath_dir in upgrade_to_fpaths.values()
        for fpath in dbutils.fs.ls(str(fpath_dir))
    ]

    # Check for any diffs between the schemas
    schema_diffs = get_schema_diffs([df.schema for df in initial_dataframes])

    # Load the data with the merged schema
    dataframes = [
        load_clean_and_add_upgrade_id(fpath.path, upgrade_id, schema_diffs=schema_diffs)
        for upgrade_id, fpath_dir in upgrade_to_fpaths.items()
        for fpath in dbutils.fs.ls(str(fpath_dir))
    ]

    united_df = reduce(partial(DataFrame.unionByName, allowMissingColumns=True), dataframes)

    return united_df


def clean_bsb_output_cols(bsb_df: DataFrame) -> DataFrame:
    """
    Cleans and subsets the columns of a DataFrame output from bsb

    Args:
    - bsb_df (DataFrame): The DataFrame to be cleaned.

    Returns:
    - DataFrame: The cleaned DataFrame

    """
    drop_cols = [
        "job_id",
        "started_at",
        "completed_at",
        "completed_status",
        "step_failures",
    ]

    return data_cleaning.edit_columns(
        df=bsb_df.drop(*drop_cols),
        remove_columns_with_substrings=[
            "report_simulation_output_emissions_",
            "report_simulation_output_system_use_",
            "report_simulation_output_include_",
            "apply_upgrade",
            "utility_bills",
            "applicable",
            "upgrade_costs",
            "add_timeseries",
            "output_format",
        ],
        remove_substrings_from_columns=["qoi_report_qoi_", "end_use_", "fuel_use_"],
        replace_column_substrings_dict={
            "report_simulation_output": "out",
            "m_btu": "energy_consumption_m_btu",
            "heat_pump_backup": "hp_bkup",
            "energy_use": "site_energy",
        },
        replace_period_character="_",
    )


def convert_column_units(bsb_df: DataFrame) -> DataFrame:
    """
    Converts the units of specified columns in a dataframe based on predefined conversion factors and suffixes.

    The function iterates through each column in the dataframe and checks if the column name ends with a specific
    suffix that indicates the unit of measurement. If a match is found, the column is renamed and its values are
    converted according to the conversion factor associated with that unit. The conversion factors and new units
    are defined for British Thermal Units (BTU) to kilowatt-hours (kWh), pounds (lb) to kilograms (kg), BTU per hour
    to kilowatts (kW), and Fahrenheit (F) to Celsius (C).

    Args:
    - bsb_df (DataFrame): The Spark DataFrame containing the columns to be converted.

    Returns:
    - DataFrame: A new DataFrame with the converted units for specified columns.
    """
    # TODO: this is quite ineffiecient and should be rewritten
    # (suffix, conversion_factor except for temp, new_unit)
    conversions = [
        ("_m_btu", constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR * 1e6, "kwh"),
        ("_lb", constants.POUND_TO_KILOGRAM, "kg"),
        ("_btu_h", constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR, "kw"),
        ("_f", 1, "c"),
    ]

    for col_name in bsb_df.columns:
        for suffix, factor, new_unit in conversions:
            if col_name.endswith(suffix):
                new_col_name = col_name.replace(suffix, f"_{new_unit}")
                conversion_expr = (
                    (F.col(new_col_name) * factor) if new_unit != "c" else ((F.col(new_col_name) - 32) * 5 / 9)
                )
                bsb_df = bsb_df.withColumnRenamed(col_name, new_col_name).withColumn(new_col_name, conversion_expr)
    return bsb_df


# This function contains all the shared preprocessing for bsb sims
def get_clean_rastock_df() -> DataFrame:
    """
    Reads, combines, cleans, and converts units of bsb simultions into single RAStock dataframe.

    Returns:
        DataFrame: A Spark DataFrame with all RAStock data.
    """
    rastock_df = read_combine_sims()
    rastock_df = clean_bsb_output_cols(rastock_df)
    rastock_df = convert_column_units(rastock_df)
    return rastock_df
