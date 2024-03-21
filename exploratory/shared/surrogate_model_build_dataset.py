# Databricks notebook source
# MAGIC %md # Build Dataset for Surrogate Model 

# COMMAND ----------

import os
import pyspark.sql.functions as F
import re

# COMMAND ----------

RESSTOCK_PATH = os.environ.get(
    "SURROGATE_MODELING_RESSTOCK_PATH",
    "gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/"
    "resstock_tmy3_release_1/",
)

BUILDING_METADATA_PARQUET_PATH = (
    RESSTOCK_PATH
    + "metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet"
)
HOURLY_OUTPUT_PATH = (
    RESSTOCK_PATH
    + "timeseries_individual_buildings/by_state/upgrade={upgrade_id}/state={state}/{building_id}-{upgrade_id}.parquet"
)
# pattern of weather files path within RESSTOCK_PATH
# examples:
# `resstock_tmy3_release_1`, `resstock_tmy3_release_1.1`:
#       `.../weather/state={state}/{geoid}_TMY3.csv`
# `resstock_amy2018_release_1`, `resstock_amy2018_release_1.1`:
#       `.../weather/state={state}/{geoid}_f018.csv`
# `comstock_amy2018_release_2`:
#       `.../weather/amy2018/{geoid}_2018.csv`
# WEATHER_FILES_PATH = RESSTOCK_PATH + 'weather/state={state}/{geoid}_TMY3.csv'

# COMMAND ----------

# MAGIC %md ## Load datasets from Parquet/CSV

# COMMAND ----------

SHARED_COLUMN_RENAME_DICT = {"bldg_id": "building_id", "upgrade": "upgrade_id"}


def clean_resstock_columns(
    df, remove_strings_from_columns=[], remove_columns_with_strings=[]
):
    """
    Clean ResStock columns by replacing '.' with an empty string in column names.
    Also remove specified strings from column names and drop columns that contain specified strings.

    Args:
      df (DataFrame): Input DataFrame
      remove_strings_from_columns (list, optional): List of strings to remove from column names. Defaults to [].
      remove_columns_with_strings (list, , optional): Remove columns that contain any of the strings in this list. Defaults to [].

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


def get_building_metadata():
    """
    Placeholder for pulling and light preprocessing of building metadata (may add type conversions). 
    """
    # Read in data and do some standard renames
    building_metadata = spark.read.parquet(
        BUILDING_METADATA_PARQUET_PATH
    ).withColumnsRenamed(SHARED_COLUMN_RENAME_DICT)

    building_metadata_cleaned = clean_resstock_columns(
        df=building_metadata,
        remove_strings_from_columns=["in__"],
        remove_columns_with_strings=[
            "simulation_control_run",
            "emissions",
            "weight",
            "applicability",
        ],
    )

    return building_metadata_cleaned


def get_annual_outputs():
    """
    Temp placeholder function for pulling and light preprocessing of annual energy consumption outputs
    Will replace with hourly (i.e., get_hourly_outputs() in datagen.py)
    """
    # Read all scenarios at once by reading baseline and all 9 upgrade files in the directory
    annual_energy_consumption_with_metadata = spark.read.parquet(
        RESSTOCK_PATH
        + "/metadata_and_annual_results/national/parquet/*_metadata_and_annual_results.parquet"
    ).withColumnsRenamed(SHARED_COLUMN_RENAME_DICT)

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


def get_weather_data():
    """
    Pull in weather data from all county TMY weather files and drop duplicated weather stations
    """
    # get any county id for each unique weather file (we only need to read in one per weather file, rest are dups)
    county_weather_station_lookup = (
        spark.read.parquet(BUILDING_METADATA_PARQUET_PATH)
        .groupby("`in.weather_file_city`")
        .agg(F.first("`in.county`").alias("county_gisjoin"))
        .withColumnRenamed("in.weather_file_city", "weather_file_city")
    )

    # pull in weather data for unique weather stataions
    # alternatively could iterate thru just unique files using
    # `for weather_city, county in county_weather_station_lookup.collect()` and then unioning, but that seems a lot slower..
    weather_data = (
        spark.read.csv(
            RESSTOCK_PATH + "weather/state=*/*_TMY3.csv", inferSchema=True, header=True
        )  # read in all county weather files
        .withColumn(
            "county_gisjoin", F.element_at(F.split(F.input_file_name(), "/|_"), -2)
        )  # get county id from filename
        .join(
            county_weather_station_lookup, on="county_gisjoin", how="inner"
        )  # subset to unique weather files
        .withColumnsRenamed(  # rename to shorter colnames
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
        .withColumn(
            "date_time", F.expr("to_timestamp(date_time)")
        )  # Add weekend column
        .withColumn(
            "weekend", F.expr("CASE WHEN dayofweek(date_time) >= 6 THEN 1 ELSE 0 END")
        )
        .withColumn(
            "datetime_formatted", F.date_format(F.col("date_time"), "MM-dd-HH:00")
        )  # Format date_time column to month-day-hour
        .drop("county_gisjoin", "date_time")
    )
    return weather_data

# COMMAND ----------

building_metadata = get_building_metadata()

# COMMAND ----------

annual_consumption = get_annual_outputs()

# COMMAND ----------

# this takes ~3 min
weather_data = get_weather_data()

# COMMAND ----------

# MAGIC %md ## Write out Delta Tables

# COMMAND ----------

table_name = "ml.surrogate_model.building_metadata"
building_metadata.write.saveAsTable(table_name, mode="overwrite")
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

table_name = "ml.surrogate_model.weather_data"
weather_data.write.saveAsTable(
    table_name, mode="overwrite", partitionBy=["weather_file_city"]
)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

table_name = "ml.surrogate_model.annual_outputs"
annual_consumption.write.saveAsTable(
    table_name, mode="overwrite", partitionBy=["upgrade_id"]
)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------


