# Databricks notebook source
# MAGIC %md # Extract Raw Dataset for Surrogate Model
# MAGIC
# MAGIC ### Goal
# MAGIC Extract and collect the raw ResStock EUSS data and RAStock data required for surrogate modeling, do some light pre-processing to prep for feature engineering, and write to a Delta Table.
# MAGIC
# MAGIC ### Process
# MAGIC * Extract and lightly preprocess various ResStock data
# MAGIC     1. building metadata
# MAGIC     2. annual outputs: read in lightly processed ResStock and RAStock independently and them combine
# MAGIC     3. hourly weather data
# MAGIC * Write each to Delta Table
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC Let `RESSTOCK_PATH = gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/resstock_tmy3_release_1`
# MAGIC - `RESSTOCK_PATH/metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet` : Parquet file of building metadata (pkey: building id)
# MAGIC - 
# MAGIC - `building_model.resstock_outputs_annual` : Lightly processed tables of outputs for ResStock 2022.1 baseline and upgrades (pkey: building id, upgrade_id, building set) 
# MAGIC - `building_model.resstock_ra_outputs_annual` : Lightly processed tables of outputs for RA Stock upgrades (pkey: building id, upgrade_id, building set) 
# MAGIC - `RESSTOCK_PATH/weather/state=*/*_TMY3.csv`: 3107 weather csvs for each county (hour [8760] x weather variable).
# MAGIC                                                Note that counties corresponding to the same weather station have identical data.
# MAGIC
# MAGIC ##### Outputs:
# MAGIC Outputs are written based on the current version number of this repo in `pyproject.toml`.
# MAGIC - `ml.surrogate_model.building_metadata_{CURRENT_VERSION_NUM}`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.building_simulation_outputs_annual_{CURRENT_VERSION_NUM}`: Annual building model simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly_{CURRENT_VERSION_NUM}`: Hourly weather data indexed by (weather_file_city, hour datetime)

# COMMAND ----------

# DBTITLE 1,Reflect changes without reimporting
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Imports
from typing import List

from cloudpathlib import CloudPath
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from src.globals import CURRENT_VERSION_NUM
from src.utils import bsb, data_cleaning
from src import feature_utils

print(CURRENT_VERSION_NUM)

# COMMAND ----------

# DBTITLE 1,Data Paths

CUBE_RAW_DATA_PATH = CloudPath("gs://the-cube") / "data" / "raw"

RESSTOCK_PATH = (
    CUBE_RAW_DATA_PATH
    / "nrel"
    / "end_use_load_profiles"
    / "2022"
    / "resstock_tmy3_release_1"
)

BUILDING_METADATA_2022_PARQUET_PATH = str(
    RESSTOCK_PATH
    / "metadata_and_annual_results"
    / "national"
    / "parquet"
    / "baseline_metadata_only.parquet"
)

BUILDING_METADATA_2024_PARQUET_PATH = str(
    CUBE_RAW_DATA_PATH
    / "bsb_sims"
    / "nrel_2024_2_baseline"
    / "baseline.parquet"
)

ANNUAL_OUTPUT_TABLE_NAME_RESSTOCK = 'building_model.resstock_outputs_annual' 
ANNUAL_OUTPUT_TABLE_NAME_RASTOCK = 'building_model.resstock_ra_outputs_annual'

HOURLY_WEATHER_CSVS_PATH = str(RESSTOCK_PATH / "weather" / "state=*/*_TMY3.csv")

# COMMAND ----------

# MAGIC %md ## Load datasets from Parquet/CSV

# COMMAND ----------

# DBTITLE 1,Functions for loading and preprocessing raw data
def extract_hourly_weather_data() -> DataFrame:
    """
    Extract and lightly preprocess weather data from all county TMY weather files:
    drop data from duplicated weather stations; subset, rename and format columns
    """
    # get any county id for each unique weather file (we only need to read in one per weather file, rest are dups)
    county_weather_station_lookup = (
        spark.read.parquet(BUILDING_METADATA_2022_PARQUET_PATH)
        .groupby("`in.weather_file_city`")
        .agg(F.first("`in.county`").alias("county_gisjoin"))
        .withColumnRenamed("in.weather_file_city", "weather_file_city")
    )

    # pull in weather data for unique weather stataions
    weather_data = (
        # read in all county weather files
        spark.read.csv(HOURLY_WEATHER_CSVS_PATH, inferSchema=True, header=True)
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
raw_building_metadata = spark.read.parquet(BUILDING_METADATA_2022_PARQUET_PATH)
building_metadata_2022 = feature_utils.clean_building_metadata(raw_building_metadata)

# COMMAND ----------

resstock2024_df = spark.read.parquet(BUILDING_METADATA_2024_PARQUET_PATH)
building_metadata_2024 = feature_utils.clean_building_metadata(resstock2024_df)

# COMMAND ----------

# Align 
from src.utils.qa_utils import compare_dataframes_string_values
# these are all deterministic so we don't need to keep
drop_columns_2024 = [
    'battery', 
    'geometry_space_combination',
    'hvac_secondary_heating_fuel', 
    'hvac_secondary_heating_partial_space_conditioning'

]
drop_columns_2022 = [
    'hvac_secondary_heating_type_and_fuel',
    'schedules'
]

building_metadata_2024 = building_metadata_2024.drop(*drop_columns_2024)

building_metadata_2022 = building_metadata_2022.drop(*drop_columns_2022)

# Appliance Features: 
# In 2024.2, the usage portion of appliance features are split into a new column and the column with
# the appliance name contains only the appliance fuel/type. Further, the usage percentage for appliances is completely determined by “Usage Level”, so there is no need to store each applaince usage levels as additional features. 
applaince_combined_type_usage_columns = [
    "clothes_dryer",
    "clothes_washer",
    "cooking_range",
    "dishwasher", 
    "refrigerator"
]
for c in applaince_combined_type_usage_columns:
    building_metadata_2022 = building_metadata_2022.withColumn(c, F.split(F.col(c), ',')[0])
    building_metadata_2024 = building_metadata_2024.drop(f"{c}_usage_level")

# Ducts: 
# In 2024 this was renamed to duct_leakage_and_insulation, and it added string "to Outside"
building_metadata_2022 = (
    building_metadata_2022
        .withColumn("duct_leakage_and_insulation", F.regexp_replace("ducts", "Leakage", "Leakage to Outside"))
        .drop('ducts')
)

# Duct and Water Heater Location: In 2024 these were expanded from just presence to more descriptive location. 
# We will keep the flag features and simply impute all those with the flag as True to an unknown token
building_metadata_2022 = (
    building_metadata_2022
        .withColumn('duct_location',
                    F.when(F.col('hvac_has_ducts') == 'Yes', "Unknown Location").otherwise("None"))
        .withColumn('water_heater_location',
            F.when(F.col('water_heater_in_unit') == 'Yes', "Inside Unit").otherwise("Outside"))
)

#In 2024 they changed misc_hot_tub_spa from 'Electric', 'Gas' -> 'Electricity', 'Natural Gas'
building_metadata_2022 = (
    building_metadata_2022
        .withColumn('misc_hot_tub_spa', 
            F.when(F.col('misc_hot_tub_spa') == "Electric", "Electricity")
            .when(F.col('misc_hot_tub_spa') == "Gas", "Natural Gas")
            .otherwise(F.col("misc_hot_tub_spa"))
        )
)

#In 2024 they changed misc_pool_heater from 'Electric', 'Gas', Solar -> 'Electricity', 'Natural Gas', 'Other Fuel'
# We're just gonna map 'Solar' to 'Other Fuel'
building_metadata_2022 = (
    building_metadata_2022
        .withColumn('misc_pool_heater', 
            F.when(F.col('misc_pool_heater') == "Electric", "Electricity")
            .when(F.col('misc_pool_heater') == "Gas", "Natural Gas")
            .when(F.col('misc_pool_heater') == "Solar", "Other Fuel")
            .otherwise(F.col('misc_pool_heater')) #should just be 
        )
)

#In 2024 they added induction as abaseline range type so they changed Electric to Electric Resistance 
building_metadata_2022 = (
    building_metadata_2022
        .withColumn('cooking_range', 
            F.when(F.col('cooking_range') == "Electric", 'Electric Resistance')
            .otherwise(F.col('cooking_range'))
        )
)

# Heat pump naming in various columns changed in 2024 to specify whether it was ducted or not, but this is just a
# deterministic function of 'hvac_has_ducts', so it is better to have fewer categories and keep aligned with 2022 naming conventions
building_metadata_2024 = (
    building_metadata_2024
        .withColumn("hvac_heating_type_and_fuel", F.regexp_replace("hvac_heating_type_and_fuel", "MSHP", "ASHP"))
        .withColumn("hvac_heating_efficiency", F.regexp_replace("hvac_heating_efficiency", "MSHP", "ASHP"))
        .withColumn(
            "hvac_cooling_efficiency",
            F.when(F.col("hvac_cooling_efficiency") == "Non-Ducted Heat Pump", "Heat Pump")
            .when(F.col("hvac_cooling_efficiency") == "Ducted Heat Pump", "Heat Pump")
            .otherwise(F.col("hvac_cooling_efficiency")),
        )
)
# However, in 2022, the names for 'hvac_heating_type' did specify ducted vs ductelss, and now in 2024 is does for 'hvac_cooling_type' too
# So lets just align with 2024 for this so that heating and cooling naming is consistent
building_metadata_2022 = (
    building_metadata_2022
        .withColumn(
                "hvac_cooling_type",
                F.when(F.col("hvac_cooling_type") == "Heat Pump", "Ducted Heat Pump")
                .otherwise(F.col("hvac_cooling_type"))
        )
    )

print("Columns in 2022 and not 2024": set(building_metadata_2022.columns).difference(building_metadata_2024.columns))
print("Columns in 2024 and not 2022": set(building_metadata_2024.columns).difference(building_metadata_2022.columns))
print("Differences between categorical values in 2022 (df1) and 2024 (df4)")
compare_dataframes_string_values(building_metadata_2022, building_metadata_2024)


# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Read annual outputs and union
#Read in outputs for ResStock upgrades
annual_resstock_outputs = spark.table(ANNUAL_OUTPUT_TABLE_NAME_RESSTOCK)
#Read in outputs for RA Stock upgrades
# RAStock only includes sims when upgrades are applicable, so this column is missing. We'll add it back with 
# True, and in the future hopefully we can have this column be output our RA Stock sims as well
annual_rastock_outputs =  spark.table(ANNUAL_OUTPUT_TABLE_NAME_RASTOCK).withColumn("applicability", F.lit(True))

# there are many columns in RAStock that are not in ResStock so these will be null for ResStock upgrades
annual_outputs = annual_rastock_outputs.unionByName(
    annual_resstock_outputs, allowMissingColumns=True
)

# cast pkeys to the right type since longs cannot be pkeys in feature tables
annual_outputs = (
    annual_outputs
        .withColumn("building_id", F.col("building_id").cast("int"))
        .withColumn("upgrade_id", F.col("upgrade_id").cast("double"))
)

# COMMAND ----------

# DBTITLE 1,Extract hourly weather data
# this takes ~3 min
hourly_weather_data = extract_hourly_weather_data()

# COMMAND ----------

# MAGIC %md ## Write out Delta Tables

# COMMAND ----------

# DBTITLE 1,Write out building metadata
table_name = f"ml.surrogate_model.building_metadata_{CURRENT_VERSION_NUM}"
building_metadata.write.saveAsTable(table_name, mode="overwrite", overwriteSchema=True)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

# DBTITLE 1,Write out annual outputs
table_name = f"ml.surrogate_model.building_simulation_outputs_annual_{CURRENT_VERSION_NUM}"
annual_outputs.write.saveAsTable(
    table_name, mode="overwrite", overwriteSchema=True, partitionBy=["upgrade_id"]
)
spark.sql(f"OPTIMIZE {table_name}")

# COMMAND ----------

# DBTITLE 1,Write out hourly weather data
table_name = f"ml.surrogate_model.weather_data_hourly_{CURRENT_VERSION_NUM}"
hourly_weather_data.write.saveAsTable(
    table_name,
    mode="overwrite",
    overwriteSchema=True,
    partitionBy=["weather_file_city"],
)
spark.sql(f"OPTIMIZE {table_name}")
