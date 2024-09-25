# Databricks notebook source
# MAGIC %md # Extract Raw Dataset for Surrogate Model
# MAGIC
# MAGIC ### Goal
# MAGIC Extract and collect the raw ResStock EUSS data required for surrogate modeling, do some light pre-processing to prep for feature engineering, and write to a Delta Table.
# MAGIC
# MAGIC ### Process
# MAGIC * Extract and lightly preprocess various ResStock data
# MAGIC     1. building metadata
# MAGIC     2. annual outputs: read in and process resstock and RAStock independently and them combine
# MAGIC     3. hourly weather data
# MAGIC * Write each to Delta Table
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC Let `RESSTOCK_PATH = gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/`
# MAGIC - `RESSTOCK_PATH/metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet` : Parquet file of building metadata (building id [550K] x building metadata variable)
# MAGIC - `RESSTOCK_PATH/metadata_and_annual_results/national/parquet/*_metadata_and_annual_results.parquet`: Parquet file of annual building model simulation outputs (building id [~550K], upgrade_id [11] x output variable)
# MAGIC - `RESSTOCK_PATH/weather/state=*/*_TMY3.csv`: 3107 weather csvs for each county (hour [8760] x weather variable).
# MAGIC                                                Note that counties corresponding to the same weather station have identical data.
# MAGIC - `gs://the-cube/data/raw/bsb_sims`: Many parquets within this folder holding all the RAStock simulations
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.building_simulation_outputs_annual`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Hourly weather data indexed by (weather_file_city, hour datetime)

# COMMAND ----------

# DBTITLE 1,Reflect changes without reimporting
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Imports
import os
from typing import List

from cloudpathlib import CloudPath
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from src import util

# COMMAND ----------

# DBTITLE 1,Data Paths
RESSTOCK_PATH = (
    CloudPath("gs://the-cube")
    / "data"
    / "raw"
    / "nrel"
    / "end_use_load_profiles"
    / "2022"
    / "resstock_tmy3_release_1"
)

BUILDING_METADATA_PARQUET_PATH = str(
    RESSTOCK_PATH
    / "metadata_and_annual_results"
    / "national"
    / "parquet"
    / "baseline_metadata_only.parquet"
)

ANNUAL_OUTPUT_PARQUET_PATH = str(
    RESSTOCK_PATH
    / "metadata_and_annual_results"
    / "national"
    / "parquet"
    / "*_metadata_and_annual_results.parquet"
)

HOURLY_WEATHER_CSVS_PATH = str(RESSTOCK_PATH / "weather" / "state=*/*_TMY3.csv")

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
    building_metadata_cleaned = util.clean_columns(
        df=building_metadata,
        remove_substrings_from_columns=["in__"],
        remove_columns_with_substrings=[
            "simulation_control_run",
            "emissions",
            "weight",
            "applicability",
            "upgrade_id",
        ],
    )

    return building_metadata_cleaned


def extract_resstock_annual_outputs() -> DataFrame:
    """
    Extract and lightly preprocess annual energy consumption outputs:
    rename and remove columns.
    """
    # Read all scenarios at once by reading baseline and all 9 upgrade files in the directory
    annual_energy_consumption_with_metadata = spark.read.parquet(
        ANNUAL_OUTPUT_PARQUET_PATH
    ).transform(transform_pkeys)

    # rename and remove columns
    annual_energy_consumption_cleaned = util.clean_columns(
        df=annual_energy_consumption_with_metadata,
        remove_substrings_from_columns=["in__", "out__", "__energy_consumption__kwh"],
        remove_columns_with_substrings=[
            # remove all "in__*" columns
            "in__",
            "emissions",
            "weight",
        ],
        replace_column_substrings_dict={"natural_gas": "methane_gas"},
    )

    return annual_energy_consumption_cleaned


#TODO: remove or flag GSHP upgrades for homes without ducts
def extract_rastock_annual_outputs() -> DataFrame:
    """
    Extract and lightly preprocess RAStock annual energy consumption outputs:
    rename and remove columns.
    """
    # TODO: add better documentation
    # # get annual outputs for all RAStock upgrades
    annual_energy_consumption_rastock = util.get_clean_rastock_df()

    # cast pkeys to the right type
    annual_energy_consumption_rastock = (
        annual_energy_consumption_rastock
            .withColumn("building_id", F.col("building_id").cast("int"))
            .withColumn("upgrade_id", F.col("upgrade_id").cast("double"))
    )

    modeled_fuel_types = ['fuel_oil', 'propane', 'electricity', 'natural_gas', 'site_energy']
    pkey_cols = ['building_id','upgrade_id']

    r_fuels = '|'.join(modeled_fuel_types)
    r_pkey = ''.join([f"(?!{k}$)" for k in pkey_cols])

    fuel_replace_dict = {f + '_': f + '__' for f in modeled_fuel_types}

    # reformat to match ResStock and do some light preprocessing 
    annual_energy_consumption_rastock_cleaned = util.clean_columns(
        df=annual_energy_consumption_rastock,
        # remove all columns unless they are
        # prefixed with "out_" followed by a modeled fuel or are a pkey
        remove_columns_with_substrings=[
            rf"^(?!out_({r_fuels})){r_pkey}.*"
        ],
        remove_substrings_from_columns=["out_", "_energy_consumption_kwh"],
        replace_column_substrings_dict={
            **fuel_replace_dict,
            **{"natural_gas": "methane_gas", 
            "permanent_spa" : "hot_tub"}}
    )

    return annual_energy_consumption_rastock_cleaned.withColumn('applicability', F.lit(True))

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
        spark.read.csv(HOURLY_WEATHER_CSVS_PATH , inferSchema=True, header=True)
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

# DBTITLE 1,Extract ResStock annual outputs
annual_resstock_outputs = extract_resstock_annual_outputs()

# COMMAND ----------

# DBTITLE 1,Extract RAStock annual outputs
annual_rastock_outputs = extract_rastock_annual_outputs()

# COMMAND ----------

# DBTITLE 1,Combine annual outputs
# there are ~25 specific end use columns in RAStock that are not in ResStock so these will be null in ResStock
annual_outputs = annual_rastock_outputs.unionByName(
    annual_resstock_outputs, allowMissingColumns=True
)

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
# TODO: move this back to the original once testing is complete
table_name = "ml.surrogate_model.building_simulation_outputs_annual_tmp"
# table_name = "ml.surrogate_model.building_simulation_outputs_annual"
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
