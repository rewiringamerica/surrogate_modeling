# Databricks notebook source
from pyspark.sql.functions import broadcast
import itertools
import math
import re
from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg

from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

!pip install tensorflow tensorflow_decision_forests
import tensorflow_decision_forests as tfdf
from tensorflow_decision_forests.keras import RandomForestModel

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, InputLayer
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredLogarithmicError

import itertools
import logging
import math
import os
import re
import calendar
from typing import Dict
import matplotlib.pyplot as plt


spark.conf.set("spark.sql.shuffle.partitions", 1536)

# COMMAND ----------

# Resstock metadata
metadata = spark.table('building_model.resstock_metadata')
eligible = ['Single-Family Detached', 'Single-Family Attached']
metadata = metadata.filter(col("in_geometry_building_type_acs").isin(eligible))
drop_list = ['in_census_division', 'in_ahs_region', 'puma_geoid', 'in_weather_file_latitude', 'in_weather_file_longitude', 'in_sqft_bin', 'in_occupants_bin', 'in_income', 'in_geometry_floor_area_bin']
metadata = metadata.drop(*drop_list)
metadata = metadata.toPandas()

# COMMAND ----------

metadata['in_hvac_backup_heating_efficiency_nominal_percent'] = 'None'
metadata['in_backup_heating_fuel'] = 'None'
met_conditions =  metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False)
metadata.loc[met_conditions, 'in_hvac_backup_heating_efficiency_nominal_percent'] = '100%'
metadata.loc[met_conditions, 'in_backup_heating_fuel'] = 'Electricity'
metadata['upgrade_id'] = 0   

# COMMAND ----------


# upgrade 1

metadata_upgrade1 = metadata.copy()
def attic_insulation_IECC_CZ1A(metadata):
    met_conditions = (metadata["in_ashrae_iecc_climate_zone_2004"] == "1A") & \
                     (metadata["in_geometry_attic_type"] == "Vented Attic") & \
                     (metadata["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13"]))
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_ceiling"] = "R-30"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def attic_insulation_IECC_CZ2A_2B_3A_3B_3C(metadata):
    met_conditions = (metadata["in_ashrae_iecc_climate_zone_2004"].isin(["2A", "2B", "3A", "3B", "3C"])) & \
                     (metadata["in_geometry_attic_type"] == "Vented Attic") & \
                     (metadata["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13", "R-19", "R-30"]))
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_ceiling"] = "R-49"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def attic_insulation_IECC_CZ4A_7C(metadata):
    met_conditions = (metadata["in_ashrae_iecc_climate_zone_2004"].isin(["4A", "4B", "4C", "5A", "5B", "6A", "6B", "7A", "7B"])) & \
                     (metadata["in_geometry_attic_type"] == "Vented Attic") & \
                     (metadata["in_insulation_ceiling"].isin(["Uninsulated", "R-7", "R-13", "R-19", "R-30", "R-38"]))
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_ceiling"] = "R-60"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def infiltration_30pct_reduction(metadata):
    met_conditions = metadata["in_infiltration_ach50"] >= 15
    if met_conditions.any():
        metadata.loc[met_conditions, "in_infiltration_ach50"] *= 0.7        
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def ducts_leakage(metadata):
    met_conditions = metadata["in_ducts_leakage"] > 0
    if met_conditions.any():
        metadata.loc[met_conditions, "in_ducts_leakage"] = 10
        metadata.loc[met_conditions, "in_ducts_insulation"] = 'R-8'
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def Drill_and_fill(metadata):
    met_conditions = metadata["in_insulation_wall"] == "Wood Stud, Uninsulated"
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_wall"] = "Wood Stud, R-13"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_01(metadata):
    metadata['upgrade_id'] = 1
    metadata["eligible_for_upgrade"] = 0
    attic_insulation_IECC_CZ1A(metadata)
    attic_insulation_IECC_CZ2A_2B_3A_3B_3C(metadata)
    attic_insulation_IECC_CZ4A_7C(metadata)
    infiltration_30pct_reduction(metadata)
    ducts_leakage(metadata)
    Drill_and_fill(metadata)
    return metadata

# Example usage:
metadata_upgrade1 = apply_upgrade_01(metadata_upgrade1)


# COMMAND ----------


## upgrade 2
metadata_upgrade2 = metadata.copy()

def apply_upgrade_foundation_wall_insulation(metadata):
    met_conditions = (metadata["in_geometry_foundation_type"].isin(["Unvented Crawlspace", "Vented Crawlspace"])) & \
                     (metadata["in_insulation_foundation_wall"] == "Uninsulated")
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_foundation_wall"] = "Wall R-10, Interior"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_foundation_wall_insulation_finished_basement(metadata):
    met_conditions = (metadata["in_geometry_foundation_type"] == "Heated Basement") & \
                     (metadata["in_insulation_foundation_wall"] == "Uninsulated")
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_foundation_wall"] = "Wall R-10, Interior"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_rim_joist_insulation(metadata):
    met_conditions = (metadata["in_geometry_foundation_type"].isin(["Unvented Crawlspace", "Vented Crawlspace", "Heated Basement"])) & \
                     (metadata["in_insulation_foundation_wall"] == "Uninsulated")
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_rim_joist"] = "R-10, Exterior"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_seal_vented_crawlspaces(metadata):
    met_conditions = (metadata["in_geometry_foundation_type"] == "Unvented Crawlspace")
    if met_conditions.any():
        metadata.loc[met_conditions, "in_geometry_foundation_type"] = "Unvented Crawlspace"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_roof_insulation_to_R30(metadata):
    met_conditions = (metadata["in_geometry_attic_type"] == "Finished Attic or Cathedral Ceilings") & \
                     (metadata["in_insulation_roof"].isin(["Finished, Uninsulated", "Finished, R-7", "Finished, R-13"]))
    if met_conditions.any():
        metadata.loc[met_conditions, "in_insulation_roof"] = "Finished, R-30"
        metadata.loc[met_conditions, "eligible_for_upgrade"] = 1

def apply_upgrade_02(metadata):
    metadata['upgrade_id'] = 2
    metadata["eligible_for_upgrade"] = 0
    attic_insulation_IECC_CZ1A(metadata)
    attic_insulation_IECC_CZ2A_2B_3A_3B_3C(metadata)
    attic_insulation_IECC_CZ4A_7C(metadata)
    infiltration_30pct_reduction(metadata)
    ducts_leakage(metadata)
    Drill_and_fill(metadata)
    apply_upgrade_foundation_wall_insulation(metadata)
    apply_upgrade_foundation_wall_insulation_finished_basement(metadata)
    apply_upgrade_rim_joist_insulation(metadata)
    apply_upgrade_seal_vented_crawlspaces(metadata)
    apply_upgrade_roof_insulation_to_R30(metadata)
    return metadata

# Example usage:
metadata_upgrade2 = apply_upgrade_02(metadata_upgrade2)


# COMMAND ----------


# Define upgrade functions
metadata_upgrade3 = metadata.copy()
metadata_upgrade4 = metadata.copy()


def apply_upgrade_03(metadata):
    metadata["eligible_for_upgrade"] = 0
    metadata['upgrade_id'] = 3
    apply_logic_asHP = ((metadata["in_hvac_has_ducts"] == True) &
                        (~metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                         metadata["in_hvac_heating_efficiency"].isin(["ASHP, SEER 10, 6.2 HSPF",
                                                                   "ASHP, SEER 13, 7.7 HSPF",
                                                                   "ASHP, SEER 15, 8.5 HSPF"])))
    
    apply_logic_msHP = ((metadata["in_hvac_has_ducts"] == False) &
                        (~metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                         metadata["in_hvac_heating_efficiency"].isin(["MSHP, SEER 14.5, 8.2 HSPF"])))
    
    metadata.loc[apply_logic_asHP, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"
    metadata.loc[apply_logic_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 15, 9.0 HSPF, Max Load"
    
    metadata.loc[apply_logic_asHP | apply_logic_msHP, "eligible_for_upgrade"] = 1
    metadata.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_cooling_type"] = 'Heat Pump'
    metadata.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_cooling_partial_space_conditioning"] = '100%'
    metadata.loc[apply_logic_asHP | apply_logic_msHP, "in_heating_fuel"] = "Electricity"
    metadata.loc[apply_logic_asHP | apply_logic_msHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
    return metadata

def apply_upgrade_04(metadata):
    metadata['upgrade_id'] = 4
    metadata["eligible_for_upgrade"] = 0
    apply_logic_ducted_msHP = (metadata["in_hvac_has_ducts"] == True)
    
    apply_logic_nonducted_msHP = ((metadata["in_hvac_has_ducts"] == False) &
                                  (~metadata["in_hvac_cooling_type"].str.contains("Heat Pump", na=False) |
                                   metadata["in_hvac_heating_efficiency"].isin(["MSHP, SEER 14.5, 8.2 HSPF",
                                                                             "MSHP, SEER 29.3, 14 HSPF, Max Load"])))
    
    metadata.loc[apply_logic_ducted_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 24, 13 HSPF"
    
    metadata.loc[apply_logic_nonducted_msHP, "in_hvac_heating_efficiency"] = "MSHP, SEER 29.3, 14 HSPF, Max Load"
    
    metadata.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "eligible_for_upgrade"] = 1
    metadata.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_cooling_type"] = 'Heat Pump'
    metadata.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_cooling_partial_space_conditioning"] = '100%'
    metadata.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_heating_fuel"] = "Electricity"
    metadata.loc[apply_logic_ducted_msHP | apply_logic_nonducted_msHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
    return metadata

# Apply upgrades

# Example usage:
metadata_upgrade3 = apply_upgrade_03(metadata_upgrade3)
metadata_upgrade4 = apply_upgrade_04(metadata_upgrade4)




# COMMAND ----------

## Upgrade 05

metadata_upgrade5 = metadata.copy()

def apply_upgrade_05(metadata):
    metadata['upgrade_id'] = 5
    apply_logic_asHP = (metadata["in_hvac_cooling_type"] == 'Heat Pump')
    
    apply_logic_else = (metadata["in_hvac_cooling_type"] != 'Heat Pump')
    
    metadata.loc[apply_logic_asHP, "in_hvac_backup_heating_efficiency_nominal_percent"] = "100%"
    metadata.loc[apply_logic_asHP, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"

    metadata.loc[apply_logic_else, "in_hvac_backup_heating_efficiency_nominal_percent"] = metadata.loc[apply_logic_else, "in_hvac_heating_efficiency_nominal_percent"]
    metadata.loc[apply_logic_else, "in_hvac_heating_efficiency"] = "ASHP, SEER 15, 9.0 HSPF"
    
    metadata.loc[apply_logic_asHP | apply_logic_else, "eligible_for_upgrade"] = 1
    metadata.loc[apply_logic_asHP | apply_logic_else, "in_hvac_cooling_type"] = 'Heat Pump'
    metadata.loc[apply_logic_asHP | apply_logic_else, "in_hvac_cooling_partial_space_conditioning"] = '100%'
    metadata['in_heating_fuel'] = "Electricity"
    metadata['in_backup_heating_fuel'] = metadata['in_heating_fuel']
    return metadata

metadata_upgrade5 = apply_upgrade_05(metadata_upgrade5)

# COMMAND ----------

metadata_w_upgrades = pd.concat([metadata, metadata_upgrade1, metadata_upgrade2, metadata_upgrade3, metadata_upgrade4, metadata_upgrade5])

# COMMAND ----------

metadata_w_upgrades

# COMMAND ----------



# Apply the function to the column
metadata_w_upgrades['in_hvac_heating_efficiency_nominal_percent'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(convert_heating_efficiency)

metadata_w_upgrades['in_hvac_seer_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_seer)

metadata_w_upgrades['in_hvac_hspf_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_hspf)

metadata_w_upgrades['in_hvac_afue_rating'] = metadata_w_upgrades['in_hvac_heating_efficiency'].apply(extract_afue)


# COMMAND ----------

metadata_w_upgrades = spark.createDataFrame(metadata_w_upgrades)

metadata_w_upgrades.write.saveAsTable("building_model.resstock_metadata_w_upgrades1_5",
                                                        mode='overwrite')
