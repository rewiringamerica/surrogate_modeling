# Databricks notebook source
from pyspark.sql.functions import broadcast
import itertools
import math
import re
from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
import re
import pandas as pd
spark.conf.set("spark.sql.shuffle.partitions", 1536)

# COMMAND ----------

# Resstock metadata loading and preprocessing
metadata = spark.table('building_model.resstock_metadata')

eligible_households = ['Single-Family Detached', 'Single-Family Attached']
metadata = metadata.filter(col("in_geometry_building_type_acs").isin(eligible_households))


## remove ineligible fuels like None and other fuel since Resstock doesn't model this
ineligible_fuels = ['Other Fuel', 'None']
metadata = (metadata.filter(~col("in_heating_fuel").isin(ineligible_fuels)))

## also remove shared cooling systems and shared heating systems (small number still left after previous filter)
metadata = (metadata.filter(col("in_hvac_cooling_type") != 'Shared Cooling'))
metadata = (metadata.filter(col("in_hvac_heating_efficiency") != 'Shared Heating'))

drop_list = ['in_census_division', 'in_ahs_region', 'puma_geoid', 'in_weather_file_latitude', 'in_weather_file_longitude', 'in_sqft_bin', 'in_occupants_bin', 'in_income', 'in_geometry_floor_area_bin']
metadata = metadata.drop(*drop_list)
#convert to pandas dataframe
metadata = metadata.toPandas()

# COMMAND ----------

## metadata feature creation
metadata['upgrade_id'] = 0   
metadata['in_hvac_backup_heating_efficiency_nominal_percent'] = 'None'
metadata['in_backup_heating_fuel'] = 'None'

met_conditions =  metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False)
metadata.loc[met_conditions, 'in_hvac_backup_heating_efficiency_nominal_percent'] = '100%'
metadata.loc[met_conditions, 'in_backup_heating_fuel'] = 'Electricity'


# COMMAND ----------



metadata_upgrade1 = metadata.copy()
def attic_insulation_IECC_CZ1A(df):
   met_conditions = (df["in_ashrae_iecc_climate_zone_2004"] == "1A") & \
                    (df["in_geometry_attic_type"] == "Vented Attic") & \
                    (df["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13"]))
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_ceiling"] = "R-30"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def attic_insulation_IECC_CZ2A_2B_3A_3B_3C(df):
   met_conditions = (df["in_ashrae_iecc_climate_zone_2004"].isin(["2A", "2B", "3A", "3B", "3C"])) & \
                    (df["in_geometry_attic_type"] == "Vented Attic") & \
                    (df["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13", "R-19", "R-30"]))
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_ceiling"] = "R-49"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def attic_insulation_IECC_CZ4A_7C(df):
   met_conditions = (df["in_ashrae_iecc_climate_zone_2004"].isin(["4A", "4B", "4C", "5A", "5B", "6A", "6B", "7A", "7B"])) & \
                    (df["in_geometry_attic_type"] == "Vented Attic") & \
                    (df["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13", "R-19", "R-30", "R-38"]))
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_ceiling"] = "R-60"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def infiltration_30pct_reduction(df):
   met_conditions = df["in_infiltration_ach50"] >= 15
   if met_conditions.any():
       df.loc[met_conditions, "in_infiltration_ach50"] *= 0.7       
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def ducts_leakage(df):
   met_conditions = df["in_ducts_leakage"] > 0
   if met_conditions.any():
       df.loc[met_conditions, "in_ducts_leakage"] = 10
       df.loc[met_conditions, "in_ducts_insulation"] = 'R-8'
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def Drill_and_fill(df):
   met_conditions = df["in_insulation_wall"] == "Wood Stud, Uninsulated"
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_wall"] = "Wood Stud, R-13"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_01(df):
    df['upgrade_id'] = 1
    df["eligible_for_upgrade"] = 0
    attic_insulation_IECC_CZ1A(df)
    attic_insulation_IECC_CZ2A_2B_3A_3B_3C(df)
    attic_insulation_IECC_CZ4A_7C(df)
    infiltration_30pct_reduction(df)
    ducts_leakage(df)
    Drill_and_fill(df)
    return df


# COMMAND ----------


def apply_upgrade_foundation_wall_insulation(df):
   met_conditions = (df["in_geometry_foundation_type"].isin(["Unvented Crawlspace", "Vented Crawlspace"])) & \
                    (df["in_insulation_foundation_wall"] == "Uninsulated")
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_foundation_wall"] = "Wall R-10, Interior"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def apply_upgrade_foundation_wall_insulation_finished_basement(df):
   met_conditions = (df["in_geometry_foundation_type"] == "Heated Basement") & \
                    (df["in_insulation_foundation_wall"] == "Uninsulated")
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_foundation_wall"] = "Wall R-10, Interior"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def apply_upgrade_rim_joist_insulation(df):
   met_conditions = (df["in_geometry_foundation_type"].isin(["Unvented Crawlspace", "Vented Crawlspace", "Heated Basement"])) & \
                    (df["in_insulation_foundation_wall"] == "Uninsulated")
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_rim_joist"] = "R-10, Exterior"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def apply_upgrade_seal_vented_crawlspaces(df):
   met_conditions = (df["in_geometry_foundation_type"] == "Unvented Crawlspace")
   if met_conditions.any():
       df.loc[met_conditions, "in_geometry_foundation_type"] = "Unvented Crawlspace"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def apply_upgrade_roof_insulation_to_R30(df):
   met_conditions = (df["in_geometry_attic_type"] == "Finished Attic or Cathedral Ceilings") & \
                    (df["in_insulation_roof"].isin(["Finished, Uninsulated", "Finished, R-7", "Finished, R-13"]))
   if met_conditions.any():
       df.loc[met_conditions, "in_insulation_roof"] = "Finished, R-30"
       df.loc[met_conditions, "eligible_for_upgrade"] = 1


def apply_upgrade_02(df):
   df['upgrade_id'] = 2
   df["eligible_for_upgrade"] = 0
   attic_insulation_IECC_CZ1A(df)
   attic_insulation_IECC_CZ2A_2B_3A_3B_3C(df)
   attic_insulation_IECC_CZ4A_7C(df)
   infiltration_30pct_reduction(df)
   ducts_leakage(df)
   Drill_and_fill(df)
   apply_upgrade_foundation_wall_insulation(df)
   apply_upgrade_foundation_wall_insulation_finished_basement(df)
   apply_upgrade_rim_joist_insulation(df)
   apply_upgrade_seal_vented_crawlspaces(df)
   apply_upgrade_roof_insulation_to_R30(df)
   return df


# COMMAND ----------


def apply_upgrade_03(df):
   df["eligible_for_upgrade"] = 0
   df['upgrade_id'] = 3
   apply_logic_asHP = ((df["in_hvac_has_ducts"] == True) &
                       (~df["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                        df["in_hvac_heating_efficiency"].isin(["ASHP, SEER 10, 6.2 HSPF",
                                                                  "ASHP, SEER 13, 7.7 HSPF",
                                                                  "ASHP, SEER 15, 8.5 HSPF"])))
  
   apply_logic_msHP = ((df["in_hvac_has_ducts"] == False) &
                       (~df["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                        df["in_hvac_heating_efficiency"].isin(["MSHP, SEER 14.5, 8.2 HSPF"])))
  
   df.loc[apply_logic_asHP, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"
   df.loc[apply_logic_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 15, 9.0 HSPF, Max Load"
  
   df.loc[apply_logic_asHP | apply_logic_msHP, "eligible_for_upgrade"] = 1
   df.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_cooling_type"] = 'Heat Pump'
   df.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_cooling_partial_space_conditioning"] = '100%'
   df.loc[apply_logic_asHP | apply_logic_msHP, "in_heating_fuel"] = "Electricity"
   df.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
   return df


def apply_upgrade_04(df):
   df['upgrade_id'] = 4
   df["eligible_for_upgrade"] = 0
   apply_logic_ducted_msHP = (df["in_hvac_has_ducts"] == True)
  
   apply_logic_nonducted_msHP = ((df["in_hvac_has_ducts"] == False) &
                                 (~df["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                                  df["in_hvac_heating_efficiency"].isin(["MSHP, SEER 14.5, 8.2 HSPF",
                                                                            "MSHP, SEER 29.3, 14 HSPF, Max Load"])))
  
   df.loc[apply_logic_ducted_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 24, 13 HSPF"
  
   df.loc[apply_logic_nonducted_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 29.3, 14 HSPF, Max Load"
  
   df.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "eligible_for_upgrade"] = 1
   df.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_cooling_type"] = 'Heat Pump'
   df.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_cooling_partial_space_conditioning"] = '100%'
   df.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_heating_fuel"] = "Electricity"
   df.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
   return df


# COMMAND ----------

## Upgrade 05


def apply_upgrade_05(df):
   df['upgrade_id'] = 5
   apply_logic_asHP = (df["in_hvac_cooling_type"] == 'Heat Pump')
  
   apply_logic_else = (df["in_hvac_cooling_type"] != 'Heat Pump')
  
   df.loc[apply_logic_asHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
   df.loc[apply_logic_asHP, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"


   df.loc[apply_logic_else, "in_hvac_backup_heating_efficiency_nominal_percent"] = df.loc[apply_logic_else, "in_hvac_heating_efficiency_nominal_percent"]
   df.loc[apply_logic_else, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"
  
   df.loc[apply_logic_asHP | apply_logic_else, "eligible_for_upgrade"] = 1
   df.loc[apply_logic_asHP | apply_logic_else, "in_hvac_cooling_type"] = 'Heat Pump'
   df.loc[apply_logic_asHP | apply_logic_else, "in_hvac_cooling_partial_space_conditioning"] = '100%'
   df['in_heating_fuel'] = "Electricity"
   df['in_backup_heating_fuel'] = df['in_heating_fuel']
   return df



# COMMAND ----------

def apply_all_upgrades(df):
    return pd.concat([apply_upgrade_01(df.copy()),
                      apply_upgrade_02(df.copy()),
                      apply_upgrade_03(df.copy()),
                      apply_upgrade_04(df.copy()),
                      apply_upgrade_05(df.copy())])


# COMMAND ----------

metadata_w_upgrades = apply_all_upgrades(metadata)

# COMMAND ----------

## preprocessing of features

metadata_w_upgrades['in_vintage'] = metadata_w_upgrades['in_vintage'].apply(vintage2age2010)

metadata_w_upgrades['in_ducts_leakage'] = data['in_ducts_leakage'].fillna(0)

metadata_w_upgrades['in_geometry_stories'] = data['in_geometry_stories'].astype(float)

metadata_w_upgrades['in_hvac_heating_efficiency_nominal_percent'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(convert_heating_efficiency)

metadata_w_upgrades['in_hvac_seer_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_seer)

metadata_w_upgrades['in_hvac_hspf_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_hspf)

metadata_w_upgrades['in_hvac_afue_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_afue)


# COMMAND ----------

## obtain cooling efficiency
met_conditions =  metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False)
metadata.loc[met_conditions, "in_hvac_cooling_efficiency"] = "SEER " + " " + metadata.loc[mask, "in_hvac_seer_rating"].astype(str)

metadata_w_upgrades["in_hvac_cooling_efficiency"] = metadata_w_upgrades["in_hvac_cooling_efficiency"].apply(extract_cooling_efficiency)


# COMMAND ----------

## Convert to SparkDF and write to directory

table_name = 'metadata_w_upgrades1_5'
database_name = 'building_model'

path = table_name + '.' + database_name

metadata_w_upgrades = spark.createDataFrame(metadata_w_upgrades)

metadata_w_upgrades.write.saveAsTable(table_name, mode='overwrite', path = path)
