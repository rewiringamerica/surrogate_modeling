# Databricks notebook source
from pyspark.sql.functions import broadcast
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
spark.conf.set("spark.sql.shuffle.partitions", 1536)


# COMMAND ----------

resstock_path = 'building_model.resstock_outputs_hourly'
weather_data_full_path = 'building_model.weather_data_yearly'
metadata_path = 'building_model.metadata_w_upgrades'

resstock = spark.table(resstock_path)
metadata = spark.table(metadata_path)
weather = spark.table(weather_data_full_path)

# COMMAND ----------

##define end uses by fuel type. And select the columns corresponding to them

heating_electric = ['out_electricity_heating_fans_pumps_energy_consumption_kwh', 'out_electricity_heating_hp_bkup_energy_consumption_kwh', 'out_electricity_heating_energy_consumption_kwh']

cooling_electric = ['out_electricity_cooling_fans_pumps_energy_consumption_kwh',
                    'out_electricity_cooling_energy_consumption_kwh']

heating_nat_gas = ['out_natural_gas_heating_hp_bkup_energy_consumption_kwh','out_natural_gas_heating_energy_consumption_kwh']

heating_fuel_oil =['out_fuel_oil_heating_hp_bkup_energy_consumption_kwh','out_fuel_oil_heating_energy_consumption_kwh']

heating_propane = ['out_propane_heating_hp_bkup_energy_consumption_kwh',
                   'out_propane_heating_energy_consumption_kwh']

# COMMAND ----------

                                                                                    
resstock = (resstock.withColumn( 
    'out_electricity_heating_total', sum(resstock[col] for col in heating_electric)).withColumn(
        'out_electricity_cooling_total', sum(resstock[col] for col in cooling_electric)).withColumn(
          'out_natural_gas_heating_total', sum(resstock[col] for col in heating_nat_gas)).withColumn(
              'out_fuel_oil_heating_total', sum(resstock[col] for col in heating_fuel_oil)).withColumn('out_propane_heating_total', sum(resstock[col] for col in heating_propane))
          ) 

drop_list = heating_electric + cooling_electric + heating_fuel_oil + heating_nat_gas + heating_propane
resstock = resstock.drop(*drop_list)

# COMMAND ----------

from pyspark.sql.functions import sum

def Create_full_data(resstock, metadata, weather, aggregation_level, table_write_path):
    if aggregation_level == 'yearly':
        resstock_yearly = (resstock).groupBy('building_id','upgrade_id').agg(
        *[sum(col).alias("sum_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']])
        
        resstock_yearly_with_metadata = (
        resstock_yearly
        .join(broadcast(metadata), on = ['building_id', 'upgrade_id']))

        resstock_yearly_with_metadata_weather = (
        resstock_yearly_with_metadata
        .join(broadcast(weather), on = ['county_geoid']))

        resstock_yearly_with_metadata_weather.write.saveAsTable(table_write_path)

    elif aggregation_level == 'monthly':
        resstock_monthly = (resstock).groupBy('building_id', 'month', 'upgrade_id').agg(
        *[sum(col).alias("sum_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']])
        
        resstock_monthly_with_metadata = (
        resstock_monthly
        .join(broadcast(metadata), on = ['building_id', 'upgrade_id']))

        resstock_monthly_with_metadata_weather = (
        resstock_monthly_with_metadata
        .join(broadcast(weather), on = ['county_geoid', 'month']))
        
        resstock_monthly_with_metadata_weather.write.saveAsTable(table_write_path)
               
    elif aggregation_level == 'daily':
        resstock_daily = (resstock).groupBy('building_id', 'day', 'month', 'upgrade_id').agg(
        *[sum(col).alias("sum_" + col) for col in resstock.columns if col not in ['building_id', 'month','upgrade_id', 'day', 'hour', 'weekday', 'timestamp']])
        
        resstock_daily_with_metadata = (
        resstock_daily
        .join(broadcast(metadata), on = ['building_id', 'upgrade_id']))

        resstock_daily_with_metadata_weather = (
        resstock_daily_with_metadata
        .join(broadcast(weather), on = ['county_geoid', 'day', 'month']))

        resstock_daily_with_metadata_weather.write.saveAsTable(table_write_path)

    else:
        raise ValueError("Only accept yearly, monthly and daily aggregation levels")


# COMMAND ----------

table_write_path = "building_model.resstock_yearly_with_metadata_weather_upgrades"

Create_full_data(resstock = resstock, metadata = metadata, weather =weather, aggregation_level = 'yearly', table_write_path = table_write_path)
