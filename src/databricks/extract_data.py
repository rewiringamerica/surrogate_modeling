# Databricks notebook source
# MAGIC %md # Extract Raw Dataset for Surrogate Model 
# MAGIC
# MAGIC ### Goal
# MAGIC Extract and collect the raw ResStock EUSS data required for surrogate modeling, do some light pre-processing to prep for feature engineering, and write to a Delta Table. 
# MAGIC
# MAGIC ### Process
# MAGIC * Extract and lightly preprocess ResStock (1) building metadata, (2) annual outputs, and (3) hourly weather data
# MAGIC * Write each to Delta Table
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs: 
# MAGIC Let `RESSTOCK_PATH = gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/`
# MAGIC - `RESSTOCK_PATH/metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet` : Parquet file of building metadata (building id [550K] x building metadata variable)
# MAGIC - `RESSTOCK_PATH/metadata_and_annual_results/national/parquet/*_metadata_and_annual_results.parquet`: Parquet file of annual building model simulation outputs (building id [~550K], upgrade_id [11] x output variable)
# MAGIC - `RESSTOCK_PATH/weather/state=*/*_TMY3.csvs`: 3107 weather csvs for each county (hour [8760] x weather variable). 
# MAGIC                                                Note that counties corresponding to the same weather station have identical data. 
# MAGIC
# MAGIC ##### Outputs: 
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.building_upgrade_simulation_outputs_annual`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Hourly weather data indexed by (weather_file_city, hour datetime)
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Outstanding
# MAGIC
# MAGIC #### Future Work
# MAGIC - Add hourly outputs
# MAGIC - Maybe do type conversion on building metadata in this script rather than downstream?

# COMMAND ----------

# DBTITLE 1,Imports
import os
import re
from typing import List

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,Data Paths
RESSTOCK_PATH = os.environ.get(
    "SURROGATE_MODELING_RESSTOCK_PATH",
    "gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/"
    "resstock_tmy3_release_1/",
)

BUILDING_METADATA_PARQUET_PATH = (
    RESSTOCK_PATH
    + "metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet"
)

ANNUAL_OUTPUT_PARQUET_PATH = (
    RESSTOCK_PATH
    + "/metadata_and_annual_results/national/parquet/*_metadata_and_annual_results.parquet"
)

# pattern of weather files path within HOURLY_WEATHER_CSVS_PATH
# examples:
# `resstock_tmy3_release_1`, `resstock_tmy3_release_1.1`:
#       `.../weather/state={state}/{county_geoid}_TMY3.csv`
HOURLY_WEATHER_CSVS_PATH = RESSTOCK_PATH + "weather/state=*/*_TMY3.csv"

# COMMAND ----------

# MAGIC %md ## Load datasets from Parquet/CSV

# COMMAND ----------

# DBTITLE 1,Functions for loading and preprocessing raw data
def transform_pkeys(df):
    return (
        df.withColumn("building_id", F.col("bldg_id").cast("int"))
        .withColumn("upgrade_id", F.col("upgrade").cast("double"))
        .drop("bldg_id", "upgrade")
    )


def clean_resstock_columns(
    df: DataFrame,
    remove_strings_from_columns: List[str],
    remove_columns_with_strings: List[str] = [],
) -> DataFrame:
    """
    Clean ResStock columns by replacing '.' with an empty string in column names.
    Also remove specified strings from column names and drop columns that contain specified strings.

    Args:
      df (DataFrame): Input DataFrame
      remove_strings_from_columns (list of str, optional): List of strings to remove from column names. Defaults to [].
      remove_columns_with_strings (list of str, , optional): Remove columns that contain any of the strings in this list. Defaults to [].

    Returns:
      DataFrame: Cleaned DataFrame
    """
    # Replace '.' with an empty string in column names so that we don't have to deal with backticks
    df = df.selectExpr(
        *[f" `{col}` as `{col.replace('.', '__')}`" for col in df.columns]
    )

    df = df.selectExpr(
        *[
            f"{col} as {re.sub('|'.join(remove_strings_from_columns), '', col)}"
            for col in df.columns
            if not re.search("|".join(remove_columns_with_strings), col)
        ]
    )
    return df


def extract_building_metadata() -> DataFrame:
    """
    Extract and lightly preprocess ResStock building metadata:
    rename and remove columns.

    Returns:
        building_metadata_cleaned (DataFrame): cleaned ResStock building metadata

    TODO: Maybe add type conversions?.
    """
    # Read in data and do some standard renames
    building_metadata = spark.read.parquet(BUILDING_METADATA_PARQUET_PATH).transform(
        transform_pkeys
    )

    # rename and remove columns
    building_metadata_cleaned = clean_resstock_columns(
        df=building_metadata,
        remove_strings_from_columns=["in__"],
        remove_columns_with_strings=[
            "simulation_control_run",
            "emissions",
            "weight",
            "applicability",
            "upgrade_id",
        ],
    )

    return building_metadata_cleaned


def extract_annual_outputs() -> DataFrame:
    """
    Extract and lightly preprocess annual energy consumption outputs from all upgrades:
    rename and remove columns.
    """
    # Read all scenarios at once by reading baseline and all 9 upgrade files in the directory
    annual_energy_consumption_with_metadata = spark.read.parquet(
        ANNUAL_OUTPUT_PARQUET_PATH
    ).transform(transform_pkeys)

    # rename and remove columns
    annual_energy_consumption_cleaned = clean_resstock_columns(
        df=annual_energy_consumption_with_metadata,
        remove_strings_from_columns=["in__", "out__", "__energy_consumption__kwh"],
        remove_columns_with_strings=[
            r"in__(?!weather_file_city)",
            "emissions",
            "weight",
        ],
    )
    return annual_energy_consumption_cleaned


def extract_hourly_weather_data():
    """
    Extract and lightly preprocess weather data from all county TMY weather files:
    drop data from duplicated weather stations; subset, rename and format columns
    """
    # get any county id for each unique weather file (we only need to read in one per weather file, rest are dups)
    county_weather_station_lookup = (
        spark.read.parquet(BUILDING_METADATA_PARQUET_PATH)
        .groupby("`in.weather_file_city`")
        .agg(F.first("`in.county`").alias("county_gisjoin"))
        .withColumnRenamed("in.weather_file_city", "weather_file_city")
    )

    # pull in weather data for unique weather stataions
    weather_data = (
        # read in all county weather files
        spark.read.csv(
            RESSTOCK_PATH + "weather/state=*/*_TMY3.csv", inferSchema=True, header=True
        )
        # get county id from filename
        .withColumn(
            "county_gisjoin", F.element_at(F.split(F.input_file_name(), "/|_"), -2)
        )
        # subset to unique weather files
        .join(county_weather_station_lookup, on="county_gisjoin", how="inner")
        # rename to shorter colnames
        .withColumnsRenamed(
            {
                "Dry Bulb Temperature [°C]": "temp_air",
                "Relative Humidity [%]": "relative_humidity",
                "Wind Speed [m/s]": "wind_speed",
                "Wind Direction [Deg]": "wind_direction",
                "Global Horizontal Radiation [W/m2]": "ghi",
                "Direct Normal Radiation [W/m2]": "dni",
                "Diffuse Horizontal Radiation [W/m2]": "diffuse_horizontal_illum",
            }
        )
        # Add weekend indicator
        .withColumn("date_time", F.expr("to_timestamp(date_time)"))
        .withColumn(
            "weekend", F.expr("CASE WHEN dayofweek(date_time) >= 6 THEN 1 ELSE 0 END")
        )
        # Format date_time column to month-day-hour
        .withColumn(
            "datetime_formatted", F.date_format(F.col("date_time"), "MM-dd-HH:00")
        )
        .drop("county_gisjoin", "date_time")
    )
    return weather_data

# COMMAND ----------

# DBTITLE 1,Extract building metadata
building_metadata = extract_building_metadata()

# COMMAND ----------

# DBTITLE 1,Extract annual outputs
annual_outputs = extract_annual_outputs()

# COMMAND ----------

# DBTITLE 1,Extract hourly weather data
# this takes ~3 min
hourly_weather_data = extract_hourly_weather_data()

# COMMAND ----------

# MAGIC %md ## Write out Delta Tables

# COMMAND ----------

# DBTITLE 1,Write out building metadata
table_name = "ml.surrogate_model.building_metadata"
building_metadata.write.saveAsTable(table_name, mode="overwrite", overwriteSchema=True)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

# DBTITLE 1,Write out annual outputs
table_name = "ml.surrogate_model.building_upgrade_simulation_outputs_annual"
annual_outputs.write.saveAsTable(
    table_name, mode="overwrite", overwriteSchema=True, partitionBy=["upgrade_id"]
)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

# DBTITLE 1,Write out hourly weather data
table_name = "ml.surrogate_model.weather_data_hourly"
hourly_weather_data.write.saveAsTable(
    table_name,
    mode="overwrite",
    overwriteSchema=True,
    partitionBy=["weather_file_city"],
)
spark.sql(f"OPTIMIZE {table_name}")
