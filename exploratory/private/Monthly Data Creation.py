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

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
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

heating_electric = ['out_electricity_heating_fans_pumps_energy_consumption_kwh', 'out_electricity_heating_hp_bkup_energy_consumption_kwh', 'out_electricity_heating_energy_consumption_kwh']

cooling_electric = ['out_electricity_cooling_fans_pumps_energy_consumption_kwh',
                    'out_electricity_cooling_energy_consumption_kwh']

heating_nat_gas = ['out_natural_gas_heating_hp_bkup_energy_consumption_kwh','out_natural_gas_heating_energy_consumption_kwh']

heating_fuel_oil =['out_fuel_oil_heating_hp_bkup_energy_consumption_kwh','out_fuel_oil_heating_energy_consumption_kwh']

heating_propane = ['out_propane_heating_hp_bkup_energy_consumption_kwh',
                   'out_propane_heating_energy_consumption_kwh']

# COMMAND ----------

resstock = spark.table('building_model.resstock_outputs_hourly')                                                                                    
resstock = (resstock.withColumn( 
    'out_electricity_heating_total', sum(resstock[col] for col in heating_electric)).withColumn(
        'out_electricity_cooling_total', sum(resstock[col] for col in cooling_electric)).withColumn(
          'out_natural_gas_heating_total', sum(resstock[col] for col in heating_nat_gas)).withColumn(
              'out_fuel_oil_heating_total', sum(resstock[col] for col in heating_fuel_oil)).withColumn('out_propane_heating_total', sum(resstock[col] for col in heating_propane))
          ) 

drop_list = heating_electric + cooling_electric + heating_fuel_oil + heating_nat_gas +heating_propane
resstock = resstock.drop(*drop_list)

# COMMAND ----------

# Resstock metadata
metadata = spark.table('building_model.resstock_metadata')
eligible = ['Single-Family Detached', 'Single-Family Attached']
metadata = metadata.filter(col("in_geometry_building_type_acs").isin(eligible))
drop_list = ['in_census_division', 'in_ahs_region', 'puma_geoid', 'in_weather_file_latitude', 'in_weather_file_longitude', 'in_sqft_bin', 'in_occupants_bin', 'in_income', 'in_geometry_floor_area_bin']
metadata = metadata.drop(*drop_list)

# COMMAND ----------




# COMMAND ----------


resstock_monthly = (resstock).groupBy('building_id', 'month','upgrade_id').agg(
    *[avg(col).alias("avg_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']]
)

#resstock_monthly = (resstock).groupBy('building_id', 'month','upgrade_id').avg().drop('avg(building_id)', 'avg(upgrade_id)', 'avg(month)','avg(day)', 'avg(hour)')

resstock_monthly_with_metadata = (
    resstock_monthly
    .join(broadcast(metadata), on = 'building_id')
)


# COMMAND ----------

weather_metadata = spark.table('building_model.weather_files_metadata')
weather_data = spark.table('building_model.weather_files_data')


weather_data = weather_data.select('temp_air', 'relative_humidity', 'wind_speed' , 'ghi',
                                                          'dni', 'dhi', 'canonical_epw_filename', 'year',
                                                          'month', 'day', 'hour')

weather_full_daily = (
    weather_data
    .join(weather_metadata.select('canonical_epw_filename', 'county_geoid'), on = 'canonical_epw_filename')
).groupBy('day','month', 'county_geoid').agg(F.max(col("temp_air")).alias('temp_high'), F.min(col("temp_air")).alias('temp_low'),
                                             F.avg(col("temp_air")).alias('temp_avg'), F.avg(col("wind_speed")).alias('wind_speed_avg'),
                                             F.avg(col("ghi")).alias('ghi_avg'),
                                             F.avg(col("dni")).alias('dni_avg'),
                                             F.avg(col("dhi")).alias('dhi_avg')
                                             )

weather_full_monthly = (
    weather_full_daily
).groupBy('month', 'county_geoid').agg(F.avg(col("temp_high")).alias('temp_high'), F.avg(col("temp_low")).alias('temp_low'),
                                             F.avg(col("temp_avg")).alias('temp_avg'), F.avg(col("wind_speed_avg")).alias('wind_speed_avg'),
                                             F.avg(col("ghi_avg")).alias('ghi_avg'),
                                             F.avg(col("dni_avg")).alias('dni_avg'),
                                             F.avg(col("dhi_avg")).alias('dhi_avg'),
                                             F.stddev(col("temp_high")).alias('std_temp_high'),
                                             F.stddev(col("temp_low")).alias('std_temp_low'),
                                             F.stddev(col("wind_speed_avg")).alias('std_wind_speed'),
                                             F.stddev(col("ghi_avg")).alias('std_ghi'),
                                             )


# COMMAND ----------

resstock_monthly_with_metadata_weather = (
    resstock_monthly_with_metadata
    .join(broadcast(weather_full_monthly), on = ['county_geoid','month'])
)

# COMMAND ----------

resstock_monthly_with_metadata_weather.write.saveAsTable("building_model.resstock_monthly_with_metadata_weather_v1")
