# Databricks notebook source
# MAGIC %md # Evaluate CNN Model

# COMMAND ----------

# MAGIC %pip install seaborn==v0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget setup
dbutils.widgets.dropdown("mode", "test", ["test", "production"])
dbutils.widgets.text("model_name", "test")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("test_size", "10000") # default in test mode
DEBUG = dbutils.widgets.get("mode") == "test"
# name of the model 
MODEL_NAME = dbutils.widgets.get("model_name")
# run ID of the model to test. If passed in by prior task in job, then overrride the input value
input_run_id = dbutils.widgets.get("run_id")
RUN_ID = dbutils.jobs.taskValues.get(taskKey = "model_training", key = "run_id", debugValue=input_run_id, default=input_run_id)
assert RUN_ID != "" "Must pass in run id-- if running in notebook, insert run id in widget text box"
# number of samples from test set to to run inference on (takes too long to run on all)
TEST_SIZE = int(dbutils.widgets.get("test_size"))
print(DEBUG)
print(MODEL_NAME)
print(RUN_ID)
print(TEST_SIZE)

# COMMAND ----------

# DBTITLE 1,Import
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC from functools import reduce
# MAGIC from typing import List
# MAGIC
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import pyspark.sql.functions as F
# MAGIC import seaborn as sns
# MAGIC from cloudpathlib import CloudPath
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.types import ArrayType, IntegerType
# MAGIC
# MAGIC from src.databricks.datagen import DataGenerator, load_data
# MAGIC from src.databricks.surrogate_model import SurrogateModel

# COMMAND ----------

# DBTITLE 1,Globals
MODEL_RUN_NAME = f"{MODEL_NAME}@{RUN_ID}"
# path to write figures to
EXPORT_FPATH = CloudPath("gs://the-cube") / "export"  # move to globals after reorg

# COMMAND ----------

# MAGIC %md ## Evaluate Model

# COMMAND ----------

# MAGIC %md ### Load Model and Inference Data

# COMMAND ----------

# DBTITLE 1,Load model
# init model
sm = SurrogateModel(name=MODEL_NAME)
# mlflow.pyfunc.get_model_dependencies(model_uri=sm.get_model_uri(run_id=RUN_ID))
# Load the unregistered model using run ID
model_loaded = mlflow.pyfunc.load_model(
    model_uri=sm.get_model_uri(run_id=RUN_ID))

# COMMAND ----------

# DBTITLE 1,Load test data
# load test data
_, _, test_data = load_data(n_test=TEST_SIZE)
# init data generator so we can get all of the features -- note that we can't use databricks automatic lookup of features since we logged with mlflow
test_gen = DataGenerator(test_data)
# reload the training set but with building id and upgrade id keys which we need (this is a little hacky..)
test_set = test_gen.init_training_set(
    train_data=test_data,
    exclude_columns=["weather_file_city"]
).load_df()
# convert from pyspark to pandas so we can run inference
inference_data = test_set.toPandas()

# COMMAND ----------

# MAGIC %md ### Run Inference

# COMMAND ----------

# DBTITLE 1,Run inference
# run inference: takes ~20s on 10,0000 and ~2m on 100,0000 samples
prediction_arr = model_loaded.predict(inference_data)

# COMMAND ----------

# DBTITLE 1,Set targets to null if fuel type is not present
# TODO: move this and the next two cells to SurrogateModelingWrapper.postprocess_result() once we add features that have indicators for presence each fuel type in the home, so this code will get a lot simpler

# list of columns containing fuel types for appliances
appliance_fuel_cols = [
    "clothes_dryer_fuel",
    "cooking_range_fuel",
    "heating_fuel",
    "hot_tub_spa_fuel",
    "pool_heater_fuel",
    "water_heater_fuel",
]

targets_formatted = np.array(
    [item.replace("_", " ").title() for item in test_gen.targets]
)
# create a N x A X M array where N is the number of test samples, A is the number of appliances that could use a particular fuel type and M is the number of fuel targets
# fuel_present_by_sample_appliance_fuel[i][j][k] = True indicates that appliance j for building sample i uses fuel k
fuel_present_by_sample_appliance_fuel = np.expand_dims(targets_formatted, axis=[
                                                       0, 1]) == np.expand_dims(inference_data[appliance_fuel_cols], 2)
# sum over appliances to just get 2d mask where fuel_present_by_sample_fuel_mask[i][k] = True indicates building sample i uses fuel k
fuel_present_by_sample_fuel_mask = fuel_present_by_sample_appliance_fuel.sum(
    1).astype(bool)
# all(ish) homes have electricity so set this to always be True
fuel_present_by_sample_fuel_mask[:, targets_formatted == "Electricity"] = True
# null out the predictions if there are no appliances with that fuel type -- we are basically setting these to 0,
# but we also want to make sure we don't give the model credit for predicting 0 when we manually set this
predictions_with_nulled_out_fuels = np.where(
    ~fuel_present_by_sample_fuel_mask, np.nan, prediction_arr)

# COMMAND ----------

# DBTITLE 1,Concatenate pkeys and predictions
# create an array of N x 2 with the pkeys needed to join this to metadata
sample_pkeys = ["building_id", "upgrade_id"]
sample_pkey_arr = inference_data[sample_pkeys].values

# combine prediction (including sum over all fuels) and pkeys to create a N x 2 (pkeys) array + M (fuel targets) + 1 (summed fuels)
targets = test_gen.targets + ["total"]
predictions_with_pkeys = np.hstack(
    [
        sample_pkey_arr,  # columns of pkeys
        predictions_with_nulled_out_fuels,  # columns for each fuel
        # column of totals summed over all fuels
        np.expand_dims(np.nansum(predictions_with_nulled_out_fuels, 1), 1),
    ]
)

# COMMAND ----------

# DBTITLE 1,Convert to prediction to pyspark df
# Create a N x M + 1 DataFrame of predictions
target_pred_labels = [f"{y}-pred" for y in targets]
pred_only = spark.createDataFrame(
    predictions_with_pkeys.tolist(),
    sample_pkeys + target_pred_labels,
).replace(float("nan"), None)

# COMMAND ----------

pred_only.display()

# COMMAND ----------

# MAGIC %md ## Post-Process Results

# COMMAND ----------

# DBTITLE 1,Combine actual and predicted targets
# Create dataframe with columns for actual and predicted values for each fuel
# and add a total columns to the actual
pred_wide = pred_only.join(
    test_data.withColumn("total", F.expr("+".join(test_gen.targets))),
    on=sample_pkeys
)

# COMMAND ----------

# DBTITLE 1,Melt to long fuel format
# Melt to long format by fuel with columns: building, upgrade, fuel, true, pred
pred_by_building_upgrade_fuel = (
    pred_wide.melt(
        ids=sample_pkeys,
        values=targets + target_pred_labels,
        valueColumnName="value",
        variableColumnName="fuel",
    )
    .withColumn(
        "target_type",
        F.when(F.split(F.col("fuel"), "-")[1] == "pred", F.lit("prediction"))
        .otherwise(F.lit("actual")),
    )
    .withColumn("fuel", F.split(F.col("fuel"), "-")[0])
    .groupBy(*sample_pkeys, "fuel")
    .pivot(pivot_col="target_type", values=["actual", "prediction"])
    .agg(F.first("value"))  # vacuous agg
)

# COMMAND ----------

# DBTITLE 1,Add and transform metadata
# add metadata that we will want to cut up results by
keep_features = ["heating_fuel", "heating_appliance_type", "ac_type"]
pred_by_building_upgrade_fuel_with_metadata = (
    test_set
    .select(*sample_pkeys, *keep_features)
    .join(pred_by_building_upgrade_fuel, on=sample_pkeys)
)

# do some manipulation on the metadata to make it more presentable and comparable to bucketed outputs
pred_by_building_upgrade_fuel_with_metadata = (
    pred_by_building_upgrade_fuel_with_metadata.withColumn(
        "heating_fuel",
        F.when(F.col("ac_type") == "Heat Pump", F.lit("Heat Pump"))
        .when(F.col("heating_fuel") == "Electricity", F.lit("Electric Resistance"))
        .when(F.col("heating_appliance_type") == "Shared", F.lit("Shared Heating"))
        .when(F.col("heating_fuel") == "None", F.lit("No Heating"))
        .otherwise(F.col("heating_fuel")),
    ).withColumn(
        "ac_type",
        F.when(F.col("ac_type") == "Shared", F.lit("Shared Cooling"))
        .when(F.col("ac_type") == "None", F.lit("No Cooling"))
        .otherwise(F.col("ac_type")),
    )
)

# COMMAND ----------

# MAGIC %md ### Calculate error metrics for upgrade and savings

# COMMAND ----------

# DBTITLE 1,Calculate error metrics
# define function to calculate absolute prediction error
@udf("double")
def APE(prediction: float, actual: float, eps=1e-3):
    """
    Calculate the Absolute Percentage Error (APE) between prediction and actual values.

    Parameters:
    - prediction (float): The predicted value.
    - actual (float): The actual value.
    - eps (float): A small value to avoid division by zero; default is 1e-3.

    Returns:
    - double: The APE value, or None if actual or pred is None or
              the actual value is within the epsilon range of zero.
    """
    if actual is None or prediction is None:
        return None
    if abs(actual) < eps:
        return None
    return abs(float(prediction - actual) / actual) * 100


# setup window to calculate savings between baseline (upgrade 0) all other upgrades
w = Window().partitionBy("building_id", "fuel").orderBy(F.asc("upgrade_id"))

# calculate baseline appliances, predicted and actual savings, and metrics (absolute error and APE) on upgrades and savings
pred_df_savings = (
    pred_by_building_upgrade_fuel_with_metadata.withColumn(
        "baseline_heating_fuel", F.first(F.col("heating_fuel")).over(w)
    )
    .withColumn("baseline_ac_type", F.first(F.col("ac_type")).over(w))
    .withColumn("prediction_baseline", F.first(F.col("prediction")).over(w))
    .withColumn("actual_baseline", F.first(F.col("actual")).over(w))
    .withColumn(
        "prediction_savings", F.col(
            "prediction_baseline") - F.col("prediction")
    )
    .withColumn("actual_savings", F.col("actual_baseline") - F.col("actual"))
    .withColumn("absolute_error", F.abs(F.col("prediction") - F.col("actual")))
    .withColumn(  # set these to Null for baseline since there is no savings
        "absolute_error_savings",
        F.when(F.col("upgrade_id") == "0", F.lit(None))
        .otherwise(F.abs(F.col("prediction_savings") - F.col("actual_savings"))),
    )
    .withColumn("absolute_percentage_error", APE(F.col("prediction"), F.col("actual")))
    .withColumn("absolute_percentage_error_savings", APE(F.col("prediction_savings"), F.col("actual_savings")))
)

# COMMAND ----------

pred_df_savings.display()

# COMMAND ----------

# MAGIC %md ### Aggregate metrics

# COMMAND ----------

# DBTITLE 1,Define function for aggregating over metrics
def aggregate_metrics(pred_df_savings: DataFrame, groupby_cols: List[str]):
    """
    Aggregates metrics for a given DataFrame by specified grouping columns.

    This function calculates the median and mean of absolute error, absolute percentage error,
    absolute error savings, and absolute percentage error savings. The results are rounded as specified.

    Parameters:
    - pred_df_savings (DataFrame): The DataFrame containing prediction savings and errors.
    - groupby_cols (list or str): A list of column names to group the DataFrame by.

    Returns:
    - DataFrame: A DataFrame aggregated by the specified groupby columns with the calculated metrics.
    """
    aggregation_expression = [
        F.round(f(F.col(colname)), round_precision).alias(
            f"{f.__name__}_{colname}")
        for f in [F.median, F.mean]
        for colname, round_precision in [
            ("absolute_error", 0),
            ("absolute_percentage_error", 1),
            ("absolute_error_savings", 0),
            ("absolute_percentage_error_savings", 1),
        ]
    ]

    return pred_df_savings.groupby(*groupby_cols).agg(*aggregation_expression)

# COMMAND ----------

# DBTITLE 1,Calculate aggregated metrics with various groupings
# all metrics are calculated on the total sum of fuels unless otherwise specified

# calculate metrics by by baseline cooling type for baseline only. showing this for all upgrades is too much,
# and probably not very useful except for maybe upgrade 1. Note that heat pumps are already covered in the heating rows below.
cooling_metrics_by_type_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings
    .where(F.col("baseline_ac_type") != "Heat Pump")
    .where(F.col("fuel") == "total")
    .where(F.col("upgrade_id") == 0),
    groupby_cols=["baseline_ac_type", "upgrade_id"],
).withColumnRenamed("baseline_ac_type", "type")

# calculate metrics by upgrade and baseline heating fuel
heating_metrics_by_type_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
    groupby_cols=["baseline_heating_fuel", "upgrade_id"],
).withColumnRenamed("baseline_heating_fuel", "type")

# calculate metrics by upgrade over all baseline types
total_metrics_by_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
    groupby_cols=["upgrade_id"],
).withColumn("type", F.lit("Total"))

# calculate metrics by fuel over all types and upgrades, skipping rows where the fuel is not present in the home
total_metrics_by_fuel = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings
        .where(F.col("prediction").isNotNull())
        .where(F.col("fuel") != "total"),
        groupby_cols=["fuel"],
    )
    .withColumn("type", F.initcap(F.regexp_replace("fuel", "_", " ")))
    .drop("fuel")
    .withColumn("upgrade_id", F.lit("Total By Fuel"))
)

# combine all of the results into a single table
dfs = [
    heating_metrics_by_type_upgrade,
    cooling_metrics_by_type_upgrade,
    total_metrics_by_upgrade,
    total_metrics_by_fuel,
]
cnn_evaluation_metrics = reduce(DataFrame.unionByName, dfs)

# COMMAND ----------

# DBTITLE 1,Display results
cnn_evaluation_metrics.display()

# COMMAND ----------

# DBTITLE 1,Save metrics table
if not DEBUG:
# save the metrics table tagged with the model name and version number
    (
        cnn_evaluation_metrics.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("userMetadata", MODEL_RUN_NAME)
        .saveAsTable("ml.surrogate_model.evaluation_metrics")
    )

# COMMAND ----------

# MAGIC %md ## Compare against Bucketed Model

# COMMAND ----------

# MAGIC %md ### Tabular Results

# COMMAND ----------

# DBTITLE 1,Read in bucketed aggregated results
# bring in bucketed metrics
bucket_metrics = pd.read_csv(
    "gs://the-cube/export/surrogate_model_metrics/bucketed_sf_hvac.csv",
    keep_default_na=False,
    dtype={"upgrade_id": "double"},
)
bucket_metrics["upgrade_id"] = bucket_metrics["upgrade_id"].astype("str")
bucket_metrics.replace({'Natural Gas': 'Methane Gas'}, inplace=True)

# COMMAND ----------

# DBTITLE 1,Combine results from both methods
cnn_metrics = cnn_evaluation_metrics.toPandas()

bucket_metrics["Model"] = "Bucketed"
cnn_metrics["Model"] = "CNN"

# rename metric columns to be more readable
metric_rename_dict = {
    "median_absolute_percentage_error_savings": "Median APE - Savings",
    "mean_absolute_percentage_error_savings": "Mean APE - Savings",
    "median_absolute_error_savings": "Median Abs Error - Savings",
    "mean_absolute_error_savings": "Mean Abs Error - Savings",
    "median_absolute_percentage_error": "Median APE",
    "mean_absolute_percentage_error": "Mean APE",
    "median_absolute_error": "Median Abs Error",
    "mean_absolute_error": "Mean Abs Error",
}

# combine and rename
metrics_combined_by_upgrade_type_model = pd.concat(
    [bucket_metrics, cnn_metrics]
).rename(columns={**metric_rename_dict, **{"upgrade_id": "Upgrade ID", "type": "Type"}})

# COMMAND ----------

# DBTITLE 1,Specify row order
# set the order in which types will appear in the table
metrics_combined_by_upgrade_type_model["Type"] = pd.Categorical(
    metrics_combined_by_upgrade_type_model["Type"],
    categories=[
        "Electricity",
        "Electric Resistance",
        "Methane Gas",
        "Propane",
        "Fuel Oil",
        "Shared Heating",
        "No Heating",
        "Heat Pump",
        "AC",
        "Room AC",
        "Shared Cooling",
        "No Cooling",
        "Total",
    ],
    ordered=True,
)

# COMMAND ----------

# DBTITLE 1,Pivot to wide on model type and sort by upgrade and type
metrics_combined_by_upgrade_type_model_metric = (
    metrics_combined_by_upgrade_type_model.melt(
        id_vars=["Type", "Model", "Upgrade ID"],
        value_vars=list(metric_rename_dict.values()),
        var_name="Metric",
    )
)

metrics_combined_by_upgrade_type = metrics_combined_by_upgrade_type_model_metric.pivot(
    index=[
        "Upgrade ID",
        "Type",
    ],
    columns=["Metric", "Model"],
    values="value",
).sort_values(["Upgrade ID", "Type"])

# COMMAND ----------

# DBTITLE 1,Display results
metrics_combined_by_upgrade_type

# COMMAND ----------

# DBTITLE 1,Write results to csv
if not DEBUG:
    metrics_combined_by_upgrade_type.to_csv(
        f"gs://the-cube/export/surrogate_model_metrics/comparison/{MODEL_RUN_NAME}_by_method_upgrade_type.csv"
    )

# COMMAND ----------

# MAGIC %md ### Visualize Comparison

# COMMAND ----------

# DBTITLE 1,Read in building level metrics for buckets
# read in data and
# set the metric to savings APE on upgrade rows and baseline APE for baseline
bucketed_pred = (
    spark.table("ml.surrogate_model.bucketed_sf_hvac_predictions")
    .withColumn(
        "absolute_percentage_error",
        F.when(F.col("upgrade_id") == 0, F.col("absolute_percentage_error"))
        .otherwise(F.col("absolute_percentage_error_savings")),
    )
    .select(
        "upgrade_id",
        F.col("baseline_appliance_fuel").alias("baseline_heating_fuel"),
        "absolute_percentage_error",
    )
)

# COMMAND ----------

# DBTITLE 1,Preprocess building level metrics for CNN
# subset to only total consumption predictions and
# set the metric to savings APE on upgrade rows and baseline APE for baseline
pred_df_savings_total = (
    pred_df_savings.where(F.col("fuel") == "total")
    .withColumn(
        "absolute_percentage_error",
        F.when(F.col("upgrade_id") == 0, F.col("absolute_percentage_error"))
        .otherwise(F.col("absolute_percentage_error_savings")),
    )
    .select("upgrade_id", "baseline_heating_fuel", "absolute_percentage_error")
)

# COMMAND ----------

# DBTITLE 1,Combine predictions from both models
# combine predictions
# do some cleanup to amke labels more presentable, including removing baseline hps for the sake of space
# convert to pandas
pred_df_savings_pd = (
    pred_df_savings_total.withColumn("Model", F.lit("CNN"))
    .unionByName(bucketed_pred.withColumn("Model", F.lit("Bucketed")))
    .replace(
        {"Shared Heating": "Shared", "No Heating": "None"},
        subset="baseline_heating_fuel",
    )
    .where(F.col("baseline_heating_fuel") != "Heat Pump")
    .withColumnsRenamed(
        {
            "baseline_heating_fuel": "Baseline Heating Fuel",
            "absolute_percentage_error": "Absolute Percentage Error",
            "upgrade_id": "Upgrade ID",
        }
    )
).toPandas()

# COMMAND ----------

# DBTITLE 1,Draw boxplot of comparison
pred_df_savings_pd_clip = pred_df_savings_pd.copy()
pred_df_savings_pd_clip["Absolute Percentage Error"] = pred_df_savings_pd_clip[
    "Absolute Percentage Error"
].clip(upper=70)

with sns.axes_style("whitegrid"):

    g = sns.catplot(
        data=pred_df_savings_pd_clip,
        x="Baseline Heating Fuel",
        y="Absolute Percentage Error",
        order=[
            "Fuel Oil",
            "Propane",
            "Natural Gas",
            "Electric Resistance",
            "Shared",
            "None",
        ],
        hue="Model",
        palette="viridis",
        fill=False,
        linewidth=1.25,
        kind="box",
        row="Upgrade ID",
        height=2.5,
        aspect=3.25,
        sharey=True,
        sharex=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "k",
            "markersize": "4",
        },
    )
g.fig.subplots_adjust(top=0.93)
g.fig.suptitle("Prediction Metric Comparison for HVAC Savings")

# COMMAND ----------

# DBTITLE 1,Function for saving fig to gcs
# TODO: move to a utils file once code is reorged
from google.cloud import storage
import io
def save_figure_to_gcfs(fig, gcspath, figure_format="png", dpi=200, transparent=False):
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
    supported_formats = ["pdf", "svg", "png", "jpg"]
    if figure_format not in supported_formats:
        raise ValueError(f"Please pass supported format in {supported_formats}")

    # Save figure image to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format=figure_format, dpi=dpi, transparent=transparent)

    # init GCS client and upload buffer contents
    client = storage.Client()
    bucket = client.get_bucket(gcspath.bucket)
    blob = bucket.blob(gcspath.blob)
    blob.upload_from_file(buf, content_type=figure_format, rewind=True)

# COMMAND ----------

# DBTITLE 1,Write out figure
if not DEBUG:
    save_figure_to_gcfs(g.fig, EXPORT_FPATH / "surrogate_model_metrics" / "comparison" / f"{MODEL_RUN_NAME}_vs_bucketed.png")
