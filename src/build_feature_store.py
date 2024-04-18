# Databricks notebook source
# MAGIC %md # Feature Store for Surrogate Model 
# MAGIC ### Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki if for access)

# COMMAND ----------

from itertools import chain
import re

import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType

from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

# MAGIC %md ## Feature Preprocessing

# COMMAND ----------

# Constants
BTU_PER_WH = 3.413

EER_CONVERSION = {
    'EER': 1.0,
    'SEER': .875,
    'SEER2': 0.91,  # ~=SEER*1.04 (https://www.marathonhvac.com/seer-to-seer2/)
    'EER2': 1.04
}

@udf(returnType=FloatType())
def extract_percentage(value:str) -> float:
    """ Extract percentage of space given

    >>> extract_percentage('100% Conditioned')
    1.0
    >>> extract_percentage('<10% Conditioned')
    0.1
    >>> extract_percentage('None')
    0.0
    >>> extract_percentage('10% Leakage, Uninsulated')
    0.1
    """
    if value == 'None':
        return 0.0
    match = re.match(r'^<?(\d+)%', value)
    try:
        return (match and float(match.group(1))) / 100.0
    except ValueError:
        raise ValueError(
            f'Cannot extract cooled space percentage from: f{value}')

@udf(returnType=IntegerType())
def vintage2age2000(vintage: str) -> int:
    """ vintage of the building in the year of 2000
    >>> vintage2age2000('<1940')
    70
    >>> vintage2age2000('1960s')
    40
    """
    vintage = vintage.strip()
    if vintage.startswith('<'):  # '<1940' bin in resstock
        return 70
    return 2000 - int(vintage[:4])

@udf(returnType=IntegerType())
def extract_r_value(construction_type: str) -> int:
    """ Extract R-value from an unformatted string

    Assumption: all baseline walls have similar R-value of ~4.
    The returned value is for additional insulation only. Examples:
        Uninsulated brick, 3w, 12": ~4 (https://ncma.org/resource/rvalues-of-multi-wythe-concrete-masonry-walls/)
        Uninsulated wood studs: ~4 (assuming 2x4 studs and 1.25/inch (air gap has higher R-value than wood), 3.5*1.25=4.375)
        Hollow Concrete Masonry Unit, Uninsulated: ~4 per 6" (https://ncma.org/resource/rvalues-ufactors-of-single-wythe-concrete-masonry-walls/)

    >>> extract_r_value('Finished, R-13')
    13
    >>> extract_r_value('Brick, 12-in, 3-wythe, R-15')
    15
    >>> extract_r_value('CMU, 6-in Hollow, Uninsulated')
    0
    >>> extract_r_value('2ft R10 Under, Horizontal')
    10
    >>> extract_r_value('R-5, Exterior')
    5
    >>> extract_r_value('Ceiling R-19')
    19
    """
    lower = construction_type.lower()
    if lower == 'none' or 'uninsulated' in lower:
        return 0
    m = re.search(r"\br-?(\d+)\b", construction_type, flags=re.I)
    if not m:
        raise ValueError(
            f'Cannot determine R-value of the construction type: '
            f'{construction_type}'
        )
    return int(m.group(1))

@udf(returnType=FloatType())
def extract_cooling_efficiency(cooling_efficiency: str) -> float:
    """ Convert a ResStock cooling efficiency into EER value

    >>> extract_cooling_efficiency('AC, SEER 13') / EER_CONVERSION['SEER']
    13.0
    >>> extract_cooling_efficiency('ASHP, SEER 20, 7.7 HSPF') / EER_CONVERSION['SEER']
    20.0
    >>> extract_cooling_efficiency('Room AC, EER 10.7')
    10.7
    >>> extract_cooling_efficiency('None') >= 99
    True
    """
    if cooling_efficiency == 'None':
        # insanely high efficiency to mimic a nonexistent cooling
        return 999.
    
    m = re.search(r"\b(SEER2|SEER|EER)\s+(\d+\.?\d*)", cooling_efficiency)
    if m:
        try:
            return EER_CONVERSION[m.group(1)] * float(m.group(2))
        except (ValueError, KeyError):
            raise ValueError(
                f'Cannot extract cooling efficiency from: {cooling_efficiency}')
    else:
        return 0.
        # raise ValueError(
        #         f'Cannot extract cooling efficiency from: {cooling_efficiency}')
            

@udf(returnType=IntegerType())
def extract_heating_efficiency(heating_efficiency: str) -> int:
    """
    "Other" IS used in single family homes, "Shared Heating" seemingly isn't
    >>> extract_heating_efficiency('Fuel Furnace, 80% AFUE')
    80
    >>> extract_heating_efficiency('ASHP, SEER 15, 8.5 HSPF')
    249
    >>> extract_heating_efficiency('None') >= 999
    True
    >>> extract_heating_efficiency('Electric Baseboard, 100% Efficiency')
    100
    >>> extract_heating_efficiency('Other')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Cannot extract heating efficiency from: ...
    """
    efficiency = heating_efficiency.rsplit(", ", 1)[-1]
    if efficiency == "None":
        return 999

    try:
        number = float(efficiency.strip().split(" ", 1)[0].strip("%"))
    except ValueError:
        raise ValueError(
            f'Cannot extract heating efficiency from: {heating_efficiency}'
        )

    if efficiency.endswith("AFUE"):
        return int(number)
    if efficiency.endswith("HSPF"):
        return int(number*100/BTU_PER_WH)

    # 'Other' - e.g. wood stove - is not supported
    return int(number)

@udf(returnType=FloatType())
def temp_from(temperature_string, base_temp = 0) -> float:
    """ Convert string Fahrenheit degrees to float F - base_temp deg

    >>> temp70('70F', base_temp = 70)
    0.0
    >>> temp70('-3F')
    -3.0
    """
    if not re.match(r"\d+F", temperature_string):
        raise ValueError(
            f"Unrecognized temperature format: {temperature_string}")
    return float(temperature_string.strip().lower()[:-1]) - base_temp

@udf(returnType=IntegerType())
def extract_window_area(value):
    """
    >>> extract_window_area('F9 B9 L9 R9')
    36
    >>> extract_window_area('F12 B12 L12 R12')
    48
    >>> extract_window_area('Uninsulated')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Cannot extract heating efficiency from: ...
    """
    try:
        return sum(int(side[1:]) for side in value.split())
    except ValueError:
        raise ValueError(f"Unrecognized format of window area: {value}")


# Make various mapping expressions
def make_map_type_from_dict(mapping):
    """
    Create a MapType mapping from a dict to pass to Pyspark
    https://stackoverflow.com/a/42983199
    """
    return F.create_map([F.lit(x) for x in chain(*mapping.items())])

yes_no_mapping = make_map_type_from_dict({'Yes' : 1, 'No' : 0})

orientation_degrees_mapping = make_map_type_from_dict(
    {
        'North': 0,
        'Northeast': 45,
        'East': 90,
        'Southeast': 135,
        'South': 180,
        'Southwest': 225,
        'West': 270,
        'Northwest': 315,
    }
)

# https://en.wikipedia.org/wiki/Luminous_efficacy 
luminous_efficacy_mapping = make_map_type_from_dict(
    {
        '100% CFL': 0.12,  # 8-15%
        '100% Incandescent': 0.02,  # 1.2-2.6%
        '100% LED': 0.15,  # 11-30%
    }
)

# COMMAND ----------

def transform_building_features():
  """
  Read and transform subset of building_metadata features for single family dettatched homes.
  Copied from _get_building_metadata() in datagen.py and translated into Pyspark. 
  """
  building_metadata_transformed = (
          spark.read.table("ml.surrogate_model.building_metadata")
          # filter to sf detatched homes with modeled heating fuel
          .where(F.col('geometry_building_type_acs') == 'Single-Family Detached') #subset to detatched until we figure out shared heating/cooling coding
          .where(F.col('hvac_heating_efficiency') != 'Other') #other fuels are not modeled in resstock
          .where(F.col('vacancy_status') == 'Occupied') #other fuels are not modeled in resstock

          # heating tranformations
          .withColumn('heating_appliance_type', F.expr("replace(hvac_heating_type_and_fuel, heating_fuel, '')")) 
          .withColumn('heating_appliance_type',  F.when(F.col('heating_appliance_type').contains('Furnace'), 'Furnace') 
                                              .when(F.col('heating_appliance_type').contains('Boiler'), 'Boiler')
                                              .when(F.col('heating_appliance_type') == '', 'None')
                                              .otherwise(F.trim(F.col('heating_appliance_type'))))
          .withColumn('heating_efficiency', extract_heating_efficiency(F.col('hvac_heating_efficiency')))
          .withColumn('heating_setpoint', temp_from(F.col('heating_setpoint'), F.lit(70)))
          .withColumn('heating_setpoint_offset_magnitude', temp_from(F.col('heating_setpoint_offset_magnitude')))
          
          # cooling tranformations
          .withColumn('ac_type', F.split(F.col('hvac_cooling_efficiency'), ',')[0])
          .withColumn('has_ac', (F.split(F.col('hvac_cooling_efficiency'), ',')[0] != 'None').cast('int'))
          .withColumn('cooled_space_proportion', extract_percentage(F.col('hvac_cooling_partial_space_conditioning')))
          .withColumn('cooling_efficiency_eer', 
                      F.when(F.col('hvac_cooling_efficiency') == 'Heat Pump', extract_cooling_efficiency(F.col('hvac_heating_efficiency')))
                             .otherwise(extract_cooling_efficiency(F.col('hvac_cooling_efficiency'))))
          .withColumn('cooling_setpoint', temp_from(F.col('cooling_setpoint'), F.lit(70)))
          .withColumn('cooling_setpoint_offset_magnitude', temp_from(F.col('cooling_setpoint_offset_magnitude')))
          
          # duct/infiltration tranformations
          .withColumn('has_ducts', yes_no_mapping[F.col('hvac_has_ducts')])
          .withColumn('ducts_insulation', extract_r_value(F.col('ducts')))
          .withColumn('ducts_leakage', extract_percentage(F.col('ducts')))
          .withColumn('infiltration_ach50', F.split(F.col('infiltration'), ' ')[0].cast('int'))

          # insulation tranformations
          .withColumn('wall_type', F.col('insulation_wall'))
          .withColumn('wall_material', F.split(F.col('wall_type'), ',')[0])
          .withColumn('insulation_wall', extract_r_value(F.col('wall_type')))
          .withColumn('insulation_slab', extract_r_value(F.col('insulation_slab')))
          .withColumn('insulation_rim_joist', extract_r_value(F.col('insulation_rim_joist')))
          .withColumn('insulation_floor', extract_r_value(F.col('insulation_floor')))
          .withColumn('insulation_ceiling_roof', F.greatest(extract_r_value(F.col('insulation_ceiling')), extract_r_value(F.col('insulation_roof'))))
          
          # misc transformations
          .withColumn('climate_zone_temp', F.substring('ashrae_iecc_climate_zone_2004', 1, 1))
          .withColumn('climate_zone_moisture', F.substring('ashrae_iecc_climate_zone_2004', 2, 1))
          .withColumn('vintage', vintage2age2000(F.col('vintage')))
          .withColumn('occupants', F.when(F.col('occupants') == '10+', 11).otherwise(F.col('occupants').cast("int")))
          .withColumn('is_vacant', (F.col('vacancy_status') != 'Occupied').cast('int'))
          .withColumn('orientation', orientation_degrees_mapping[F.col('orientation')])
          .withColumn('window_area', extract_window_area(F.col('window_areas')))
          #.withColumn('lighting_efficiency', luminous_efficacy_mapping[F.col('lighting')])

          # subset to all possible features of interest (will expand in future)
          .select(
              # primary keys
              'building_id', 
              'upgrade_id', 
              # foreign key
              'weather_file_city',
              # heating
              'heating_fuel', 
              'heating_appliance_type',
              'heating_efficiency',
              'heating_setpoint',
              'heating_setpoint_offset_magnitude',
              # cooling
              'ac_type',
              'has_ac',
              'cooled_space_proportion',
              'cooling_efficiency_eer',
              'cooling_setpoint',
              'cooling_setpoint_offset_magnitude',
              # ducts
              'has_ducts',
              'ducts_insulation',
              'ducts_leakage',
              'infiltration_ach50',
              # insulalation
              'wall_material',
              'insulation_wall',
              'insulation_slab',
              'insulation_rim_joist',
              'insulation_floor',
              'insulation_ceiling_roof',
              # misc
              F.col('bedrooms').cast('int'), 
              F.col('geometry_stories').cast('int').alias('stories'), 
              F.col('geometry_foundation_type').alias('foundation_type'),
              F.col('geometry_attic_type').alias('attic_type'),
              'climate_zone_temp', 
              'climate_zone_moisture',
              'sqft',
              'vintage',
              'is_vacant',
              'occupants',
              'orientation',
              'window_area',
              #'lighting_efficiency'
          )
  )
  return building_metadata_transformed
    

def transform_weather_features():
  """
  Read and transform weather timeseries table, 
  storing all weather ts as a 8670 len array within each weather feature as a column
  """
  weather_df =  spark.read.table("ml.surrogate_model.weather_data")
  weather_pkeys = ["weather_file_city"]

  weather_data_arrays = (
      weather_df
        .groupBy(weather_pkeys).agg(
          *[F.collect_list(c).alias(c) for c in weather_df.columns if c not in weather_pkeys]
        )
  )
  return weather_data_arrays


# COMMAND ----------

weather_data_transformed = transform_weather_features()
building_metadata_transformed = transform_building_features()

# COMMAND ----------

building_metadata_transformed.groupby('climate_zone_temp', 'climate_zone_moisture').count().display()

# COMMAND ----------

# MAGIC %md ## Create new schema in catalog in the Unity Catalog MetaStore
# MAGIC
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.
# MAGIC To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- if you do not see `ml` listed here, this means you do not have permissions
# MAGIC SHOW CATALOGS

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS surrogate_model;
# MAGIC USE SCHEMA surrogate_model;

# COMMAND ----------

# MAGIC %md ## Create the feature table
# MAGIC
# MAGIC See [API reference for GCP](https://docs.gcp.databricks.com/machine-learning/feature-store/python-api.html).
# MAGIC
# MAGIC TODO: Once features are mostly set, modify to update features rather than create/drop following [tutorial](https://docs.databricks.com/en/_extras/notebooks/source/machine-learning/feature-store-with-uc-taxi-example.html)

# COMMAND ----------

# create a FeatureEngineeringClient.
fe = FeatureEngineeringClient()

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS ml.surrogate_model.building_metadata_features;
# DROP TABLE IF EXISTS ml.surrogate_model.weather_features

# COMMAND ----------

# fe.create_table(
#     name="ml.surrogate_model.building_metadata_features",
#     primary_keys=["building_id", "upgrade_id"],
#     df=building_metadata_transformed,
#     schema=building_metadata_transformed.schema,
#     description="building metadata features",
# )

fe.write_table(
    name="ml.surrogate_model.building_metadata_features",
    df=building_metadata_transformed,
    mode="merge",
)

# fe.create_table(
#     name="ml.surrogate_model.weather_features",
#     primary_keys=["weather_file_city"],
#     df=weather_data_transformed,
#     schema=weather_data_transformed.schema,
#     description="weather timeseries array features",
# )

# fe.write_table(
#     name="ml.surrogate_model.building_metadata_features",
#     df=weather_data_transformed,
#     mode="merge",
# )

# COMMAND ----------

