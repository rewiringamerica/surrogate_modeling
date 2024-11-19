# Databricks notebook source
# MAGIC %md
# MAGIC # Process sampled resstock data (from v3.3.0) into same format as 2022.1 file

# COMMAND ----------

dbutils.widgets.text("n_sample_tag", "10k")

# COMMAND ----------

from itertools import chain
import sys

import pyspark.sql.functions as F
from pyspark.sql.types import StringType

sys.path.append("../../src")
from dmutilslocal import sumo

# COMMAND ----------

N_SAMPLE_TAG = dbutils.widgets.get("n_sample_tag")

# COMMAND ----------

# from dohyo import county_gisjoin_to_geoid

# Import is not working :  ModuleNotFoundError: No module named 'dohyo' when trying to apply fn later on

# copying function here from dohyo.py TODO: sort out imports
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

# on gcs: buildstock with 10k, 1M, 2M, & 5M rows

# raw input
RESSTOCK_SAMPLED_DATA_PATH = f"gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/buildstock_{N_SAMPLE_TAG}.csv"


# GCS bucket we'll write to
RESSTOCK_SAMPLED_2022FORMATTED_PATH = (
    f"gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/building_metadata_2022format_{N_SAMPLE_TAG}.snappy.parquet"
)

# COMMAND ----------

def process_raw_to_match_2022format(input_path):
    """
    Function to process resstock building metadata from 2024.2 format (resstock v 3.3.0) to 2022.1 format. Surrogate model was developed on 2022.1 data.

    Parameters
    ----------
        input_path: the GCS bucket that contains sampled building metadata from resstock
            example: 'gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/buildstock_10k.csv'

    Returns
    -------
        spark dataframe
    """
    # read into spark df for typing (keeping everything object)
    df = spark.read.option("header", True).csv(input_path)

    ## Process data

    # LOWERCASE & UNDERSCORE
    df = df.select([F.col(c).alias("_".join(c.lower().split())) for c in df.columns])

    # RENAME
    if "upgrade" not in df.columns:
        sdf = df.withColumn("upgrade", F.lit(0))

    # corrections to specific columns
    df = df.withColumnsRenamed(
        {
            "building": "bldg_id",
            "income_recs2015": "income_recs_2015",
            "income_recs2020": "income_recs_2020",
            "ashrae_iecc_climate_zone_2004_-_2a_split": "ashrae_iecc_climate_zone_2004_2_a_split",
            "duct_leakage_and_insulation": "ducts",
        }
    )

    ## replace values to match 2022 format

    # called just "Leakage" in 2022
    df = df.withColumn("ducts", F.regexp_replace("ducts", "Leakage to Outside", "Leakage"))
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
    # Align 'other fuel' from 2024 to match 2024
    # 1: "heating_fuel": "Wood" is considered an other fuel in 2022 and this will get filtered out downstream anyway
    # 2: "misc_pool_heater", "Solar" exists only in 2022 and "Other Fuel" exists only in 2024, so just convert Other to Solar :shrug:
    # 3: "misc_hot_tub_spa": Other Fuel" exists only in 2024, so just convert it to Electricity since
    #     since all homes have already have electricity as a valid guel (as opposed to methane gas)
    df = (
        df.withColumn(
            "heating_fuel", F.when(F.col("heating_fuel") == "Wood", "Other Fuel").otherwise(F.col("heating_fuel"))  # 1
        )
        .withColumn(
            "misc_pool_heater",  # 2
            F.when(F.col("misc_pool_heater") == "Other Fuel", "Solar").otherwise(F.col("misc_pool_heater")),
        )
        .withColumn(
            "misc_hot_tub_spa",  # 3
            F.when(F.col("misc_hot_tub_spa") == "Other Fuel", "Electricity").otherwise(F.col("misc_hot_tub_spa")),
        )
    )

    # In 2024 electric ranges are split into induction and resistance
    df = df.withColumn(
        "cooking_range",
        F.when(F.col("cooking_range").contains("Electric"), "Electricity").otherwise(F.col("cooking_range")),
    )

    # there are no unvented attics in 2022-- the insulation offered by an unvented attic is most
    # similar to a Finished Attic/Cathedral Ceiling, but this is only possible for homes with >1 story
    def fix_unvented_attic(attic_type, stories):
        return (
            F.when((attic_type == "Unvented Attic") & (stories == 1), "Vented Attic")
            .when((attic_type == "Unvented Attic") & (stories > 1), "Finished Attic or Cathedral Ceilings")
            .otherwise(attic_type)
        )

    df = df.withColumn(
        "geometry_attic_type", fix_unvented_attic(F.col("geometry_attic_type"), F.col("geometry_stories_low_rise"))
    )

    # combine columns that are split in 2024 release & drop older
    combine_cols = [
        ("clothes_dryer", "clothes_dryer_usage_level"),
        ("clothes_washer", "clothes_washer_usage_level"),
        ("cooking_range", "cooking_range_usage_level"),
        ("dishwasher", "dishwasher_usage_level"),
        ("refrigerator", "refrigerator_usage_level"),
    ]
    for (utility, usage_level) in combine_cols:
        df = df.withColumn(utility, F.concat_ws(", ", F.col(utility), F.col(usage_level))).drop(usage_level)

    # add a "sqft" field that is the midpoint
    # TODO: fix this to be not the midpoint
    # Define the mapping
    sqft_midpoint_mapping = {
        "0-499": 250,
        "500-749": 625,
        "750-999": 875,
        "1000-1499": 1250,
        "1500-1999": 1750,
        "2000-2499": 2250,
        "2500-2999": 2750,
        "3000-3999": 3500,
        "4000+": 8000,
    }
    # Create a mapping expression
    mapping_expr = F.create_map([F.lit(x) for x in chain(*sqft_midpoint_mapping.items())])
    # map the values
    df = df.withColumn("sqft", mapping_expr[F.col("geometry_floor_area")])

    ## Set weather city
    # county -> weather city lookup table
    geographies_df = spark.table("geographic_crosswalk.resstock_county_to_other_geographies").select(
        "county_geoid",
        F.col("in_weather_file_city").alias("weather_file_city"),
        F.col("in_weather_file_latitude").alias("weather_file_latitude"),
        F.col("in_weather_file_longitude").alias("weather_file_longitude"),
    )

    # extract county gisjoin from county_and_puma and inner join on county_geoid
    # NOTE: buildings in counties not in RessStock 2022.1 (e.g., HI) will be dropped
    df = (
        df.withColumn("county", F.split(F.col("county_and_puma"), ",")[0])
        .withColumn("county_geoid", county_gisjoin_to_geoid(F.col("county")))
        .join(geographies_df, on="county_geoid", how="inner")
    )

    return df


# COMMAND ----------

# DBTITLE 1,Processing
metadata_2022_format = process_raw_to_match_2022format(RESSTOCK_SAMPLED_DATA_PATH)

# COMMAND ----------

# metadata_2022_format.count()  # just under expected number since buildngs in HI got dropped

# COMMAND ----------

# DBTITLE 1,Write out files if desired
# resulting file should be renamed & saved as gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/sampled_building_metadata_2022format_*/building_metadata_*.snappy.parquet
# metadata_2022_format.repartition(1).write.option("compression", "snappy").mode("overwrite").parquet(RESSTOCK_SAMPLED_2022FORMATTED_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Extracting building metadata
# MAGIC As in [this link](https://github.com/rewiringamerica/surrogate_modeling/blob/916364b47a81b775f88b0aaa3ccdacc9d3a46eac/scripts/extract_data.py#L225). Relying on functions defined in src/dmutilslocal (for now)

# COMMAND ----------

# if reading from file:
# BUILDING_METADATA_PARQUET_PATH = f"gs://the-cube/data/processed/sampling_resstock/resstock_v3.3.0/sampled_building_metadata_2022format_{N_SAMPLE_TAG}/building_metadata_1M.snappy.parquet"
#  metadata_2022_format = spark.read.parquet(BUILDING_METADATA_PARQUET_PATH)

# COMMAND ----------

building_metadata = sumo.clean_building_metadata(metadata_2022_format)

# COMMAND ----------

# building_metadata.count() # fewer than before since removed home types that are not of interest

# COMMAND ----------

# DBTITLE 1,write out sampled building metadata
table_name = f"ml.megastock.building_metadata_{N_SAMPLE_TAG}"
building_metadata.write.saveAsTable(table_name, mode="overwrite", overwriteSchema=True)
spark.sql(f"OPTIMIZE {table_name}")
