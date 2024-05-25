# Databricks notebook source
# MAGIC %md # Evaluate CNN Model
# MAGIC

# COMMAND ----------

# !pip install seaborn==v0.13.0
# dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, IntegerType

from src.databricks.datagen import DataGenerator, load_data
from src.databricks.model import Model

# COMMAND ----------

import mlflow

# COMMAND ----------

_, _, test_data = load_data(n_subset=500)
model = Model(name="test")

# # for right now have to limit the test set since driver seems to be running out of mem
# target_test_size = 75000
# target_n_building_frac = target_test_size / test_data.count()
# test_building_id_subset = (
#     test_data.select("building_id").distinct().sample(target_n_building_frac)
# )
# test_data_sub = test_data.join(test_building_id_subset, on="building_id")
# print(test_data_sub.count())
# # score using  latest registered model
mlflow.pyfunc.get_model_dependencies(model.get_model_uri())
# pred_df = model.score_batch(test_data=test_data, run_id = '0260c2961eed47baa6a8d2919e986ba8')
pred_df = model.score_batch(test_data=test_data)
pred_df.display()


# COMMAND ----------



# COMMAND ----------

# MODEL_NAME = 'sf_hvac_by_fuel'
# MODEL_TESTSET_PREDICTIONS_TABLE = f'ml.surrogate_model.{MODEL_NAME}_predictions'
# MODEL_VERSION_NUMBER = spark.sql(f"SELECT userMetadata FROM (DESCRIBE HISTORY {MODEL_TESTSET_PREDICTIONS_TABLE }) ORDER BY version DESC LIMIT 1").rdd.map(lambda x: x['userMetadata']).collect()[0]
# MODEL_VERSION_NAME = f'ml.surrogate_model.{MODEL_NAME}@v{MODEL_VERSION_NUMBER}'

# COMMAND ----------

# targets = ['electricity', 'fuel_oil', 'natural_gas', 'propane'] #in theory could prob pull this from model artifacts..
# pred_df = spark.table(MODEL_TESTSET_PREDICTIONS_TABLE)
# building_features = spark.table('ml.surrogate_model.building_features')

# pred_df = pred_df.join(building_features, on = ['building_id', 'upgrade_id'])

# COMMAND ----------

targets = DataGenerator.targets

# COMMAND ----------

# Define the UDF function
@udf(ArrayType(IntegerType()))
def update_array(values, cond1, cond2):
    # Initialize result array with current values
    result = values[:]
    # Check conditions and update specific array positions
    if cond1 == 'A':
        result[0] = 0  # Set first element to 0 if condition1 is 'A'
    if cond2 == 'B':
        result[1] = 0  # Set second element to 0 if condition2 is 'B'
    # Add more conditions as needed
    return result

pred_df.where(F.c)

# COMMAND ----------

@udf("array<double>")
def APE(prediction, actual, eps = 1E3):
    return [abs(float(x - y))/y*100 if y > eps else None for x, y in zip(prediction, actual) ]

# COMMAND ----------

w = Window().partitionBy('building_id').orderBy(F.asc('upgrade_id'))

def element_wise_subtract(a, b):
    return F.expr(f"transform(arrays_zip({a}, {b}), x -> abs(x.{a} - x.{b}))")


pred_df_hvac_process =  (
    pred_df
        .replace("None", 'No Heating', subset = 'heating_fuel')
        .replace("None", 'No Cooling', subset = 'ac_type')
        .withColumn('heating_fuel', 
            F.when(F.col('ac_type') == 'Heat Pump', F.lit("Heat Pump"))
            .when(F.col("heating_fuel") == 'Electricity', F.lit("Electric Resistance"))
            .when(F.col("heating_appliance_type") == 'Shared', F.lit("Shared Heating"))
            .when(F.col("heating_fuel") == 'None', F.lit("No Heating"))
            .otherwise(F.col('heating_fuel')))
        .withColumn('ac_type',
            F.when(F.col("ac_type") == "Shared", F.lit("Shared Cooling"))
            .when(F.col('ac_type') == 'None', F.lit("No Cooling"))
            .otherwise(F.col('ac_type')))
        .withColumn('total', F.expr('+'.join(targets)))
        .withColumn("actual", F.array(targets + ['total']))
        .withColumn('prediction', F.array_insert("prediction", F.size(F.col('prediction'))+1, F.aggregate("prediction", F.lit(0.0), lambda acc, x: acc + x)))
)

pred_df_savings = (
    pred_df_hvac_process
        .withColumn('baseline_heating_fuel', F.first(F.col('heating_fuel')).over(w))
        .withColumn('baseline_ac_type', F.first(F.col('ac_type')).over(w))

        .withColumn('prediction_baseline', F.first(F.col('prediction')).over(w))
        .withColumn('actual_baseline', F.first(F.col('actual')).over(w))

        .withColumn('prediction_savings', element_wise_subtract('prediction', 'prediction_baseline'))
        .withColumn('actual_savings', element_wise_subtract('actual', 'actual_baseline'))

        .withColumn('absolute_error', element_wise_subtract('prediction', 'actual'))
        .withColumn('absolute_percentage_error', APE(F.col('prediction'), F.col('actual')))
        .withColumn('absolute_error_savings', element_wise_subtract('prediction_savings', 'actual_savings'))
        .withColumn('absolute_percentage_error_savings', APE(F.col('prediction_savings'), F.col('actual_savings')))
)


# COMMAND ----------

pred_df_savings.display()

# COMMAND ----------

def aggregate_metrics(pred_df_savings, groupby_cols, target_idx):
    
    aggregation_expression = [
        f(F.col(c)[target_idx]).alias(f"{f.__name__}_{c}")
        for f in [F.median, F.mean] 
        for c in ['absolute_percentage_error_savings', 'absolute_error_savings', 'absolute_percentage_error','absolute_error']
    ]

    return (
        pred_df_savings
            .groupby(*groupby_cols)
            .agg(*aggregation_expression)
        )

# COMMAND ----------


heating_metrics_by_type_upgrade = (aggregate_metrics(
    pred_df_savings = pred_df_savings,
        groupby_cols=['baseline_heating_fuel', 'upgrade_id'],
        target_idx=4)
        #target_idx=0)
    .withColumnRenamed('baseline_heating_fuel', 'type')
    .withColumn('category', F.lit('heating'))
)

cooling_metrics_by_type_upgrade = (
    aggregate_metrics(
        pred_df_savings = pred_df_savings.where(F.col('baseline_ac_type') != 'Heat Pump'),
        groupby_cols=['baseline_ac_type', 'upgrade_id'],
        target_idx=4)
         #target_idx=1)
    .withColumnRenamed('baseline_ac_type', 'type')
    .withColumn('category', F.lit('cooling'))
)

total_metrics_by_upgrade = (
    aggregate_metrics(
        pred_df_savings = pred_df_savings,
        groupby_cols=['upgrade_id'],
        target_idx=4)
    .withColumn('type', F.lit('Total'))
    .withColumn('category', F.lit('total'))
)

cnn_evaluation_metrics = heating_metrics_by_type_upgrade.unionByName(cooling_metrics_by_type_upgrade).unionByName(total_metrics_by_upgrade)

# COMMAND ----------

cnn_evaluation_metrics.display()

# COMMAND ----------

# save the metrics table tagged with the model name and version number
(cnn_evaluation_metrics
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("userMetadata", MODEL_VERSION_NAME)
    .saveAsTable('ml.surrogate_model.evaluation_metrics')
)

# COMMAND ----------

# MAGIC %md ## Compare against Bucketed Model

# COMMAND ----------

#bring in bucketed metrics and compare against that. 

bucket_metrics = pd.read_csv(
    'gs://the-cube/export/surrogate_model_metrics/bucketed_sf_hvac.csv',
    keep_default_na=False,
    dtype = {'upgrade_id' : 'str'})
cnn_metrics = cnn_evaluation_metrics.toPandas()

# COMMAND ----------

bucket_metrics['Model'] = 'Bucketed'
cnn_metrics['Model'] = 'CNN'

# COMMAND ----------

metric_rename_dict = {
    'median_absolute_percentage_error_savings' : 'Median APE - Savings', 
    'mean_absolute_percentage_error_savings': 'Mean APE - Savings', 
    'median_absolute_error_savings': 'Median Abs Error - Savings', 
    'mean_absolute_error_savings' : 'Mean Abs Error - Savings', 
    'median_absolute_percentage_error' : 'Median APE', 
    'mean_absolute_percentage_error' : 'Mean APE',
    'median_absolute_error' : 'Median Abs Error',
    'mean_absolute_error' : 'Mean Abs Error',
}

# COMMAND ----------

metrics_combined = (
    pd.concat([bucket_metrics, cnn_metrics])
        .rename(columns={**metric_rename_dict, **{'upgrade_id' : 'Upgrade ID', 'type': 'Type'}})
        .melt(
            id_vars = ['Type',  'Model', 'Upgrade ID', 'category'], 
            value_vars=list(metric_rename_dict.values()),
            var_name='Metric'
    )
)

metrics_combined['Type'] = pd.Categorical(
    metrics_combined['Type'], 
    categories=[
        'Electric Resistance', 'Natural Gas','Propane', 'Fuel Oil','Shared Heating','No Heating',
        'Heat Pump','AC', 'Room AC',  'Shared Cooling','No Cooling', 
        'Total'],
    ordered=True
)

metrics_combined = (
        metrics_combined
            .pivot(
                index = ['Upgrade ID', 'category', 'Type',],
                columns = ['Metric',  'Model'], 
                values = 'value')
            .droplevel('category')
)


# COMMAND ----------

metrics_combined

# COMMAND ----------

metrics_combined.to_csv(f'gs://the-cube/export/surrogate_model_metrics/comparison/{MODEL_VERSION_NAME}_by_method_upgrade_type.csv', float_format = '%.2f')

# COMMAND ----------

# MAGIC %md ## Visualize Comparison between Model Metrics

# COMMAND ----------

pred_df_savings_hvac = (
    pred_df_savings
        .withColumn('absolute_percentage_error',
            F.when(F.col('upgrade_id') == 0, F.col('absolute_percentage_error')[4])
            .otherwise( F.col('absolute_percentage_error_savings')[4])
        )
        .select('upgrade_id', 'baseline_heating_fuel', 'absolute_percentage_error')
) 

# COMMAND ----------

bucketed_pred  = (
    spark.table('ml.surrogate_model.bucketed_sf_hvac_predictions')
        .withColumn('absolute_percentage_error',
            F.when(F.col('upgrade_id') == 0, F.col('absolute_percentage_error'))
            .otherwise( F.col('absolute_percentage_error_savings')))
        .select('upgrade_id', F.col('baseline_appliance_fuel').alias('baseline_heating_fuel'), 'absolute_percentage_error')
)

# COMMAND ----------

pred_df_savings_pd = (
    pred_df_savings_hvac
        .withColumn('Model', F.lit('CNN'))
        .unionByName(bucketed_pred.withColumn('Model', F.lit('Bucketed')))
        .withColumnsRenamed(
            {'baseline_heating_fuel' : 'Baseline Heating Fuel', 
             "absolute_percentage_error" : "Absolute Percentage Error", 
             'upgrade_id' : 'Upgrade ID'})
    ).toPandas()

# COMMAND ----------

pred_df_savings_pd_clip = pred_df_savings_pd.copy()
pred_df_savings_pd_clip['Absolute Percentage Error'] =pred_df_savings_pd_clip['Absolute Percentage Error'].clip(upper = 70)

with sns.axes_style("whitegrid"):

    g = sns.catplot(
        data = pred_df_savings_pd_clip, x='Baseline Heating Fuel', y="Absolute Percentage Error",
        order = ['Fuel Oil', 'Propane', 'Natural Gas', 'Electric Resistance',  'Heat Pump', 'Shared Heating', 'No Heating'],
        hue = 'Model', palette = 'viridis',  fill=False,  linewidth=1.25,
        kind = 'box',  row="Upgrade ID", row_order = ['0', '1', '3', '4'], height=3, aspect= 3, sharey=True, sharex=True,
        showfliers=False, showmeans=True, meanprops={'marker':'o','markerfacecolor':'white','markeredgecolor':'k','markersize':'4'}, 
    )
g.fig.subplots_adjust(top=.93)
g.fig.suptitle('Prediction Metric Comparison for HVAC Savings')


# COMMAND ----------

#TODO: move to a utils file once code is reorged 
import io
from google.cloud import storage
from cloudpathlib import CloudPath

def save_figure_to_gcfs(fig, gcspath, figure_format ='png', dpi = 200, transparent=False):
    """ 
    Write out a figure to google cloud storage

    Args:
        fig (matplotlib.figure.Figure): figure object to write out
        gcspath (cloudpathlib.gs.gspath.GSPath): filepath to write to in GCFS
        figure_format (str): file format in ['pdf', 'svg', 'png', 'jpg']. Defaults to 'png'.
        dpi (int): resolution in dots per inch. Only relevant if format non-vector ('png', 'jpg'). Defaults to 200. 

    Returns:
        pyspark.sql.dataframe.DataFrame
    
    Modified from source: 
    https://stackoverflow.com/questions/54223769/writing-figure-to-google-cloud-storage-instead-of-local-drive
    """
    supported_formats = ['pdf', 'svg', 'png', 'jpg']
    if figure_format not in supported_formats:
        raise ValueError(f"Please pass supported format in {supported_formats}")

    # Save figure image to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format=figure_format, dpi = dpi, transparent=transparent)

    # init GCS client and upload buffer contents
    client = storage.Client()
    bucket = client.get_bucket(gcspath.bucket)
    blob = bucket.blob(gcspath.blob)  
    blob.upload_from_file(buf, content_type=figure_format, rewind=True)

# COMMAND ----------

save_figure_to_gcfs(g.fig, CloudPath('gs://the-cube') / 'export'/ 'surrogate_model_metrics' / 'comparison'/f'{MODEL_VERSION_NAME}_vs_bucketed.png')

# COMMAND ----------


