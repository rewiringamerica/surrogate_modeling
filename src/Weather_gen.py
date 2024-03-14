# Databricks notebook source
from pyspark.sql.functions import broadcast
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
spark.conf.set("spark.sql.shuffle.partitions", 1536)


# COMMAND ----------

#load data
weather_metadata_path = 'building_model.weather_files_metadata'
weather_data_path = 'building_model.weather_files_data'
weather_metadata = spark.table(weather_metadata_path)
weather_data = spark.table(weather_data_path)



# COMMAND ----------

# select relevant features
relevant_features = ['temp_air', 'relative_humidity', 'wind_speed' , 'ghi',
                                                          'dni', 'dhi', 'canonical_epw_filename', 'year',
                                                          'month', 'day', 'hour']
weather_data = weather_data.select(relevant_features)

# convert celcius to farenheit

weather_data = weather_data.withColumn("temp_air", (F.col("temp_air") * (9/5)) + 32)

# create features for CDD, HDD, heat pump switch threshold, and freezing point.
weather_data = weather_data.withColumn(
    "below_32", F.when(F.col("temp_air") < 32, 1).otherwise(0)
).withColumn(
    "below_41", F.when(F.col("temp_air") < 41, 1).otherwise(0)
).withColumn(
    "HDD", F.when(F.col("temp_air") > 65, F.col("temp_air") - 65).otherwise(0)
).withColumn(
    "CDD", F.when(F.col("temp_air") < 65, 65 - F.col("temp_air")).otherwise(0)
)

# COMMAND ----------

def WeatherAggregation(weather_data = weather_data , weather_metadata = weather_metadata, aggregation_level = 'yearly'):
    '''Create weather data for each county geoid for the given Time aggregation.
    
    Creates select aggregations for the selected features over the given temporal aggregation level. Aggregations
    mostly consist of averages and standard deviations, where the standard deviations. We also compute the number of HDD and CDD as well 

     Args:
      weather_data: A spark dataframe containing weather features 
      weather_metadata: A spark dataframe containing information mapping weather files to county_geoid
      aggregation_level: String which can take values 'yearly' 'monthly', and 'daily' 

  Returns:
      A spark dataframe for the given temporal aggregation.
    '''
    weather_full_daily = (
    weather_data
    .join(weather_metadata.select('canonical_epw_filename', 'county_geoid'), on = 'canonical_epw_filename')
    ).groupBy('day','month', 'county_geoid').agg(F.max(col("temp_air")).alias('temp_high'), F.min(col("temp_air")).alias('temp_low'),
                                             F.avg(col("temp_air")).alias('temp_avg'), F.avg(col("wind_speed")).alias('wind_speed_avg'),
                                             F.avg(col("ghi")).alias('ghi_avg'),
                                             F.avg(col("dni")).alias('dni_avg'),
                                             F.avg(col("dhi")).alias('dhi_avg'),
                                             F.avg(col('HDD')).alias('HDD'),
                                             F.avg(col('CDD')).alias('CDD'),
                                             F.sum(col('below_32')).alias('below_32'),
                                             F.sum(col('below_41')).alias('below_41'),
                                             )
    if aggregation_level == 'daily':
        return weather_full_daily
    if aggregation_level == 'monthly':
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
                                             F.sum(col('HDD')).alias('HDD'),
                                             F.sum(col('CDD')).alias('CDD'),
                                             F.sum(col('below_41')).alias('below_41'),
                                             F.sum(col('below_32')).alias('below_32'),
                                             )
        return weather_full_monthly
    if aggregation_level == 'yearly':
        weather_full_yearly = (
        weather_full_daily
        ).groupBy('county_geoid').agg(F.avg(col("temp_high")).alias('temp_high'), F.avg(col("temp_low")).alias('temp_low'),
                                             F.avg(col("temp_avg")).alias('temp_avg'), F.avg(col("wind_speed_avg")).alias('wind_speed_avg'),
                                             F.avg(col("ghi_avg")).alias('ghi_avg'),
                                             F.avg(col("dni_avg")).alias('dni_avg'),
                                             F.avg(col("dhi_avg")).alias('dhi_avg'),
                                             F.stddev(col("temp_high")).alias('std_temp_high'),
                                             F.stddev(col("temp_low")).alias('std_temp_low'),
                                             F.stddev(col("wind_speed_avg")).alias('std_wind_speed'),
                                             F.stddev(col("ghi_avg")).alias('std_ghi'),
                                             F.sum(col('HDD')).alias('HDD'),
                                             F.sum(col('CDD')).alias('CDD'),
                                             F.sum(col('below_41')).alias('below_41'),
                                             F.sum(col('below_32')).alias('below_32'),
                                             )
        return weather_full_yearly

# COMMAND ----------

aggregation_level = 'yearly'

weather_data_full = WeatherAggregation(weather_data = weather_data , weather_metadata = weather_metadata, aggregation_level = aggregation_level)

table_name = 'weather_data_full_' + aggregation_level
database_name = 'building_model'

path = table_name + '.' + database_name

(weather_data_full.write.saveAsTable(
  table_name,
  format='delta',
  mode='overwrite', 
  overwriteSchema = True,
  path=path)
  )
