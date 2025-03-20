# Databricks notebook source
# MAGIC %md
# MAGIC # Process sampled resstock data (from v3.3.0)

# COMMAND ----------

dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from src.globals import CURRENT_VERSION_NUM, MEGASTOCK_BUILDING_METADATA_TABLE
from src import feature_utils

# COMMAND ----------

# get number of samples to use
N_SAMPLE_TAG = dbutils.widgets.get("n_sample_tag").lower()

# Raw samples filepath
RESSTOCK_SAMPLED_DATA_PATH = f"gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/buildstock_{N_SAMPLE_TAG}.csv"

# COMMAND ----------

@udf(StringType())
def county_gisjoin_to_geoid(gisjoin: str) -> str:
    # Ensure GISJOIN starts with 'G' and is the correct length
    if not gisjoin.startswith("G") or len(gisjoin) != 8:
        raise ValueError("Invalid County GISJOIN format")
    # Extract the state FIPS (characters 1-2) and county FIPS (characters 4-6)
    state_fips = gisjoin[1:3]  # Characters 1-2 (after 'G')
    county_fips = gisjoin[4:7]  # Characters 4-6
    # Combine to form the county geoid
    county_geoid = f"{state_fips}{county_fips}"
    return county_geoid


# COMMAND ----------

def process_raw_to_match_energy_plus_format(input_path: str) -> DataFrame:
    """
    Function to process resstock building metadata from 2024.2 format (resstock v3.3.0) to the the same
    format as EnergyPlus outputs. 

    Parameters
    ----------
        input_path: the GCS bucket that contains sampled building metadata from resstock
            example: 'gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/buildstock_10k.csv'

    Returns
    -------
        
    """
    # read into spark df for typing (keeping everything object)
    df = spark.read.option("header", True).csv(input_path)

    ## Process data

    # LOWERCASE & UNDERSCORE
    df = df.select([F.col(c).alias("_".join(c.lower().split())) for c in df.columns])

    # RENAME
    if "upgrade" not in df.columns:
        sdf = df.withColumn("upgrade", F.lit(0))

    # naming corrections to specific columns
    df = df.withColumnsRenamed(
        {
            "building": "bldg_id",
            "income_recs2015": "income_recs_2015",
            "income_recs2020": "income_recs_2020",
            "ashrae_iecc_climate_zone_2004_-_2a_split": "ashrae_iecc_climate_zone_2004_2_a_split",
        }
    )

    # Align 'other fuel' "heating_fuel": "Wood" was mapped to "Other Fuel" in the sampler so we need to map it to "Other Fuel"
    # so that it gets filtered out when we remove homes with "heating_fuel"="Other Fuel" downstream
    df = df.withColumn("heating_fuel", F.when(F.col("heating_fuel") == "Wood", "Other Fuel").otherwise(F.col("heating_fuel")))

    # match 2022 format in this case since this is a simpler representation
    # heat pump category names are the same for ducted and ductless (minisplit) 2022
    df = (
        df.withColumn("hvac_heating_type_and_fuel", F.regexp_replace("hvac_heating_type_and_fuel", "MSHP", "ASHP"))
        .withColumn("hvac_heating_efficiency", F.regexp_replace("hvac_heating_efficiency", "MSHP", "ASHP"))
        .withColumn(
            "hvac_cooling_efficiency",
            F.when(F.col("hvac_cooling_efficiency") == "Non-Ducted Heat Pump", "Heat Pump")
            .when(F.col("hvac_cooling_efficiency") == "Ducted Heat Pump", "Heat Pump")
            .otherwise(F.col("hvac_cooling_efficiency")),
        )
    )

    # add a "sqft" field that is the midpoint for each bin and housing data based on AHS
    sqftage_mapping = spark.createDataFrame(pd.read_csv("sqftage_mapping.csv"))
    df = df.join(sqftage_mapping, on=["geometry_floor_area", "geometry_building_type_acs"], how="left")

    ## Set weather city
    # county -> weather city lookup table
    geographies_df = spark.table("geographic_crosswalk.resstock_county_to_other_geographies").select(
        "county_geoid",
        F.col("in_weather_file_city").alias("weather_file_city"),
        F.col("in_weather_file_latitude").alias("weather_file_latitude"),
        F.col("in_weather_file_longitude").alias("weather_file_longitude"),
    )

    # extract county gisjoin from county_and_puma and inner join on county_geoid
    # NOTE: buildings in counties not in RessStock 2022.1 (e.g., AK, HI) will be dropped
    df = (
        df.withColumn("county", F.split(F.col("county_and_puma"), ",")[0])
        .withColumn("county_geoid", county_gisjoin_to_geoid(F.col("county")))
        .join(geographies_df, on="county_geoid", how="inner")
    )

    return df


# COMMAND ----------

# DBTITLE 1,Processing
metadata_formatted = process_raw_to_match_energy_plus_format(RESSTOCK_SAMPLED_DATA_PATH)

# COMMAND ----------

metadata_formatted.count()  # just under expected number since buildings in AK and HI got dropped

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extracting building metadata

# COMMAND ----------

megastock_metadata = feature_utils.clean_building_metadata(metadata_formatted).withColumn("building_set", F.lit('MegaStock'))

# COMMAND ----------

# building_metadata.count() # fewer than before since removed home types that are not of interest

# COMMAND ----------

# DBTITLE 1,write out sampled building metadata
table_name = f"{MEGASTOCK_BUILDING_METADATA_TABLE}_{N_SAMPLE_TAG}_{CURRENT_VERSION_NUM}"
print(table_name)
megastock_metadata.write.saveAsTable(table_name, mode="overwrite", overwriteSchema=True)
spark.sql(f"OPTIMIZE {table_name}")
