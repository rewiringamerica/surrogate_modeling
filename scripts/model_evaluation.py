# Databricks notebook source
# MAGIC %md # Evaluate CNN Model

# COMMAND ----------

# MAGIC %pip install mlflow==2.13.0 seaborn==v0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from src import versioning

# get current poetry version of surrogate model repo to tag tables with
CURRENT_VERSION = versioning.get_poetry_version_no()

# COMMAND ----------

# DBTITLE 1,Widget setup
dbutils.widgets.dropdown("mode", "test", ["test", "production"])
dbutils.widgets.text("model_name", CURRENT_VERSION)
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("test_size", "10000")  # default in test mode

# COMMAND ----------

DEBUG = dbutils.widgets.get("mode") == "test"
# name of the model
MODEL_NAME = dbutils.widgets.get("model_name")
# run ID of the model to test. If passed in by prior task in job, then overrride the input value
input_run_id = dbutils.widgets.get("run_id")
RUN_ID = dbutils.jobs.taskValues.get(
    taskKey="model_training",
    key="run_id",
    debugValue=input_run_id,
    default=input_run_id,
)
assert (
    RUN_ID != ""
), "Must pass in run id-- if running in notebook, insert run id in widget text box"
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
# MAGIC from pyspark.sql import DataFrame, Column
# MAGIC from pyspark.sql.window import Window
# MAGIC
# MAGIC from src.datagen import DataGenerator, load_data
# MAGIC from src.surrogate_model import SurrogateModel

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
model_loaded = mlflow.pyfunc.load_model(model_uri=sm.get_model_uri(run_id=RUN_ID))

# COMMAND ----------

# DBTITLE 1,Load test data
# load test data
_, _, test_data = load_data(n_test=TEST_SIZE)
# init data generator so we can get all of the features -- note that we can't use databricks automatic lookup of features since we logged with mlflow
test_gen = DataGenerator(test_data)
# reload the training set but with building id and upgrade id keys which we need (this is a little hacky..)
test_set = test_gen.init_training_set(
    train_data=test_data, exclude_columns=["weather_file_city"]
).load_df()
# convert from pyspark to pandas so we can run inference
inference_data = test_set.toPandas()

# COMMAND ----------

# MAGIC %md ### Run Inference

# COMMAND ----------

# DBTITLE 1,Run inference
# run inference: output a N x M matrix of predictions
# where N is the number of rows in the input data table
# and M is the number of target columns
# takes ~20s on 10,0000 and ~2m on 100,0000 samples
prediction_arr = model_loaded.predict(inference_data)

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
        prediction_arr,  # columns for each fuel
        # column of totals summed over all fuels
        np.expand_dims(np.nansum(prediction_arr, 1), 1),
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

# MAGIC %md ## Combine actual, predictions, and bucketed predictions (benchmark), and calculate savings

# COMMAND ----------

# DBTITLE 1,Combine actual and predicted targets
# Create dataframe with columns for actual and predicted values for each fuel
# and add a total columns to the actual
pred_wide = pred_only.join(
    test_data.withColumn("total", F.expr("+".join(test_gen.targets))), on=sample_pkeys
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
        F.when(F.split(F.col("fuel"), "-")[1] == "pred", F.lit("prediction")).otherwise(
            F.lit("actual")
        ),
    )
    .withColumn("fuel", F.split(F.col("fuel"), "-")[0])
    .groupBy(*sample_pkeys, "fuel")
    .pivot(pivot_col="target_type", values=["actual", "prediction"])
    .agg(F.first("value"))  # vacuous agg
)

# COMMAND ----------

# DBTITLE 1,Calculate savings
# setup window to calculate savings between baseline (upgrade 0) all other upgrades
w_building = Window().partitionBy("building_id", "fuel").orderBy(F.asc("upgrade_id"))

# calculate savings
pred_by_building_upgrade_fuel_savings = (
    pred_by_building_upgrade_fuel.withColumn(
        "prediction_baseline", F.first(F.col("prediction")).over(w_building)
    )
    .withColumn("actual_baseline", F.first(F.col("actual")).over(w_building))
    .withColumn("actual_savings", F.col("actual_baseline") - F.col("actual"))
    .withColumn(
        "prediction_savings", F.col("prediction_baseline") - F.col("prediction")
    )
    .withColumn("prediction_arr", F.array("prediction", "prediction_savings"))
)

# COMMAND ----------

# DBTITLE 1,Load bucketed predictions
# read in bucketed predictions for consumption and savings, and absolute errors in the case of baseline buckets
bucketed_pred = (
    spark.table("ml.surrogate_model.bucketed_sf_predictions")
    .withColumn(
        "prediction_arr_bucketed",
        F.array("kwh_upgrade_median", "kwh_delta_median", "absolute_error"),
    )
    .select("building_id", "upgrade_id", "prediction_arr_bucketed")
    .withColumn("fuel", F.lit("total"))
)

# COMMAND ----------

# DBTITLE 1,Combine actual and both model predictions
# join the bucketed predictions and melt into long format so that "model" is a column
# note that we have to do some fiddling with arrays to make the melts work
pred_by_building_upgrade_fuel_model = (
    pred_by_building_upgrade_fuel_savings.join(
        bucketed_pred, on=["upgrade_id", "building_id", "fuel"], how="left"
    )
    .melt(
        ids=[
            "upgrade_id",
            "building_id",
            "fuel",
            "actual_baseline",
            "prediction_baseline",
            "actual",
            "actual_savings",
        ],
        values=["prediction_arr", "prediction_arr_bucketed"],
        valueColumnName="prediction_arr",
        variableColumnName="model",
    )
    .withColumn(
        "model",
        F.when(F.col("model") == "prediction_arr", F.lit("Surrogate")).otherwise(
            F.lit("Bucketed")
        ),
    )
    .withColumn("prediction", F.col("prediction_arr")[0])
    .withColumn("prediction_savings", F.col("prediction_arr")[1])
    .withColumn(
        "absolute_error", F.col("prediction_arr")[2]
    )  # only non-null for baseline bucketed pred
    .drop("prediction_arr")
)

# COMMAND ----------

# DBTITLE 1,Add and transform metadata
# add metadata that we will want to cut up results by
baseline_appliance_features = [
    "heating_fuel",
    "heating_appliance_type",
    "ac_type",
    "water_heater_fuel",
    "clothes_dryer_fuel",
    "cooking_range_fuel",
    # "is_mobile_home",
    # "is_attached",
    # "unit_level_in_building"
]
pred_by_building_upgrade_fuel_model_with_metadata = test_set.select(
    *sample_pkeys, *baseline_appliance_features
).join(pred_by_building_upgrade_fuel_model, on=sample_pkeys)

# do some manipulation on the metadata to make it more presentable and comparable to bucketed outputs
pred_by_building_upgrade_fuel_model_with_metadata = (
    pred_by_building_upgrade_fuel_model_with_metadata.withColumn(
        "heating_fuel",
        F.when(F.col("ac_type") == "Heat Pump", F.lit("Heat Pump"))
        .when(F.col("heating_fuel") == "Electricity", F.lit("Electric Resistance"))
        .when(F.col("heating_fuel") == "None", F.lit("No Heating"))
        .otherwise(F.col("heating_fuel")),
    ).withColumn(
        "ac_type",
        F.when(F.col("ac_type") == "None", F.lit("No Cooling")).otherwise(
            F.col("ac_type")
        ),
    )
)

# COMMAND ----------

# MAGIC %md ### Calculate error metrics for upgrade and savings

# COMMAND ----------

# DBTITLE 1,Calculate error metrics

# define function to calculate absolute prediction error
@udf("double")
def APE(abs_error: float, actual: float, eps=1e-3):
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
    if abs_error is None:
        return None
    if abs(actual) < eps:
        return None
    return abs(abs_error / actual) * 100


# 1. Set baseline appliances
# 2. Calculate absolute error, with handling some special cases for bucketed predictions:
#   * for baseline, we should pull the errors that were previously computed in another script
#   * all other upgrade should be set to null we only predict consumption by end use in buckets, which are not comparable
# 3. Calcuate savings absolute errror, setting these to Null for baseline since there is no savings
# 4. Calculate APE for consumption and savings
pred_df_savings = (
    pred_by_building_upgrade_fuel_model_with_metadata.withColumn(  # 1
        "baseline_appliance",
        F.when(
            F.col("upgrade_id") == 6,
            F.first(F.col("water_heater_fuel")).over(w_building),
        )
        .when(
            F.col("upgrade_id") == 8.1,
            F.first(F.col("clothes_dryer_fuel")).over(w_building),
        )
        .when(
            F.col("upgrade_id") == 8.2,
            F.first(F.col("cooking_range_fuel")).over(w_building),
        )
        .otherwise(F.first(F.col("heating_fuel")).over(w_building)),
    )
    .withColumn("baseline_ac_type", F.first(F.col("ac_type")).over(w_building))
    .withColumn(  # 2
        "absolute_error",
        F.when(
            F.col("model") == "Bucketed",
            F.when(F.col("upgrade_id") == 0, F.col("absolute_error")).otherwise(
                F.lit(None)
            ),
        ).otherwise(F.abs(F.col("prediction") - F.col("actual"))),
    )
    .withColumn(  # 3
        "absolute_error_savings",
        F.when(F.col("upgrade_id") == 0, F.lit(None)).otherwise(
            F.abs(F.col("prediction_savings") - F.col("actual_savings"))
        ),
    )
    .withColumn(
        "absolute_percentage_error", APE(F.col("absolute_error"), F.col("actual"))
    )  # 4
    .withColumn(
        "absolute_percentage_error_savings",
        APE(F.col("absolute_error_savings"), F.col("actual_savings")),
    )
)

# COMMAND ----------

# MAGIC %md ### Aggregate metrics

# COMMAND ----------

# DBTITLE 1,Define function for aggregating over metrics

# define function to calculate absolute prediction error
def wMAPE(abs_error_col: Column, actual_col: Column) -> Column:
    """
    Calculate the weighted Mean Absolute Percentage Error (wMAPE) on a pyspark df.

    Parameters:
    - abs_error_col (float): The absolute error for a sample
    - actual_col (float): The actual value for a sample.

    Returns:
    - double: wMAPE value over a group
    """
    return F.sum(abs_error_col) / F.sum(F.abs(actual_col))


def aggregate_metrics(pred_df_savings: DataFrame, groupby_cols: List[str]):
    """
    Aggregates metrics for a given DataFrame by specified grouping columns.

    This function calculates the median and mean of absolute error, absolute percentage error,
    absolute error savings, and absolute percentage error savings. It also calculated weighted
    mean absolute percentage error for consumption and savings. The results are rounded as specified.

    Parameters:
    - pred_df_savings (DataFrame): The DataFrame containing prediction savings and errors.
    - groupby_cols (list or str): A list of column names to group the DataFrame by.

    Returns:
    - DataFrame: A DataFrame aggregated by the specified groupby columns with the calculated metrics.
    """
    aggregation_expression = [
        F.round(f(F.col(colname)), round_precision).alias(f"{f.__name__}_{colname}")
        for f in [F.median, F.mean]
        for colname, round_precision in [
            ("absolute_error", 0),
            ("absolute_percentage_error", 1),
            ("absolute_error_savings", 0),
            ("absolute_percentage_error_savings", 1),
        ]
    ]

    aggregation_expression += [
        F.round(100 * wMAPE(F.col("absolute_error"), F.col("actual")), 1).alias(
            "weighted_mean_absolute_percentage_error"
        ),
        F.round(
            100 * wMAPE(F.col("absolute_error_savings"), F.col("actual_savings")), 1
        ).alias("weighted_mean_absolute_percentage_error_savings"),
    ]

    return pred_df_savings.groupby(*groupby_cols).agg(*aggregation_expression)


# COMMAND ----------

# DBTITLE 1,Calculate aggregated metrics with various groupings
# all metrics are calculated on the total sum of fuels unless otherwise specified

# calculate metrics by by baseline cooling type for baseline only. showing this for all upgrades is too much,
# and probably not very useful except for maybe upgrade 1. Note that heat pumps are already covered in the heating rows below.
cooling_metrics_by_type_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("baseline_ac_type") != "Heat Pump")
    .where(F.col("fuel") == "total")
    .where(F.col("upgrade_id") == 0),
    groupby_cols=["baseline_ac_type", "upgrade_id", "model"],
).withColumnRenamed("baseline_ac_type", "type")

# calculate metrics by upgrade and baseline heating fuel
heating_metrics_by_type_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
    groupby_cols=["baseline_appliance", "upgrade_id", "model"],
).withColumnRenamed("baseline_appliance", "type")

# calculate metrics by upgrade over all baseline types
total_metrics_by_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
    groupby_cols=["upgrade_id", "model"],
).withColumn("type", F.lit("Total"))

# calculate metrics by fuel over all types and upgrades, skipping rows where the fuel is not present in the home
total_metrics_by_fuel = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(F.col("model") == "Surrogate")
        .where(F.col("prediction").isNotNull())
        .where(F.col("fuel") != "total"),
        groupby_cols=["fuel", "model"],
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
metrics_by_upgrade_type_model = reduce(DataFrame.unionByName, dfs)

# COMMAND ----------

# DBTITLE 1,Save metrics table
if not DEBUG:
    # save the metrics table tagged with the model name and version number
    (
        metrics_by_upgrade_type_model.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("userMetadata", MODEL_RUN_NAME)
        .saveAsTable("ml.surrogate_model.evaluation_metrics")
    )

# COMMAND ----------

# MAGIC %md ## Format for export

# COMMAND ----------

# DBTITLE 1,Convert to pandas
metrics_by_upgrade_type_model_pd = metrics_by_upgrade_type_model.toPandas()

# COMMAND ----------

# DBTITLE 1,Rename columns for readabiity
# rename metric columns to be more readable
metric_rename_dict = {
    "median_absolute_percentage_error_savings": "Median APE - Savings",
    "mean_absolute_percentage_error_savings": "Mean APE - Savings",
    "median_absolute_error_savings": "Median Abs Error - Savings",
    "mean_absolute_error_savings": "Mean Abs Error - Savings",
    "weighted_mean_absolute_percentage_error_savings": "Weighted Mean APE - Savings",
    "median_absolute_percentage_error": "Median APE",
    "mean_absolute_percentage_error": "Mean APE",
    "median_absolute_error": "Median Abs Error",
    "mean_absolute_error": "Mean Abs Error",
    "weighted_mean_absolute_percentage_error": "Weighted Mean APE",
}

key_rename_dict = {"upgrade_id": "Upgrade ID", "type": "Type", "model": "Model"}

metrics_by_upgrade_type_model_pd.rename(
    columns={**metric_rename_dict, **key_rename_dict}, inplace=True
)

# COMMAND ----------

# DBTITLE 1,Specify row order
# set the order in which types will appear in the table
metrics_by_upgrade_type_model_pd["Type"] = pd.Categorical(
    metrics_by_upgrade_type_model_pd["Type"],
    categories=[
        "Electricity",
        "Electric Resistance",
        "Methane Gas",
        "Propane",
        "Fuel Oil",
        "No Heating",
        "None",
        "Heat Pump",
        "AC",
        "Room AC",
        "No Cooling",
        "Total",
    ],
    ordered=True,
)

# COMMAND ----------

# DBTITLE 1,Pivot to wide on model type and sort by upgrade and type
metrics_by_upgrade_type_model_metric = metrics_by_upgrade_type_model_pd.sort_values('Model').melt(
    id_vars=["Type", "Model", "Upgrade ID"],
    value_vars=list(metric_rename_dict.values()),
    var_name="Metric",
)

metrics_by_upgrade_type = metrics_by_upgrade_type_model_metric.pivot(
    index=[
        "Upgrade ID",
        "Type",
    ],
    columns=["Metric", "Model"],
    values="value",
).sort_values(["Upgrade ID", "Type"])

# COMMAND ----------

# DBTITLE 1,Display Results
metrics_by_upgrade_type

# COMMAND ----------

# DBTITLE 1,Write results to csv
if not DEBUG:
    metrics_by_upgrade_type.to_csv(
        f"gs://the-cube/export/surrogate_model_metrics/comparison/{MODEL_RUN_NAME}_by_upgrade_type.csv"
    )

# COMMAND ----------

# MAGIC %md ### Visualize Comparison

# COMMAND ----------

# DBTITLE 1,Preprocess building level metrics
# subset to only total consumption predictions and
# set the metric to abs error on savings on upgrade rows and abs error for baseline
pred_df_savings_total = (
    pred_df_savings.where(F.col("fuel") == "total")
    .withColumn(
        "absolute_error",
        F.when(F.col("upgrade_id") == 0, F.col("absolute_error")).otherwise(
            F.col("absolute_error_savings")
        ),
    )
    .select(
        "upgrade_id",
        "baseline_appliance",
        "absolute_error",
        F.col("model").alias("Model"),
    )
)

# COMMAND ----------

# DBTITLE 1,Label cleanup
# do some cleanup to make labels more presentable, including removing baseline hps and shared heating for the sake of space
pred_df_savings_pd = (
    pred_df_savings_total.replace(
        {"No Heating": "None", "Electric Resistance": "Electricity"},
        subset="baseline_appliance",
    )
    .where(F.col("baseline_appliance") != "Heat Pump")
    .withColumnsRenamed(
        {
            "baseline_appliance": "Baseline Fuel",
            "absolute_error": "Absolute Error (kWh)",
            "upgrade_id": "Upgrade ID",
        }
    )
).toPandas()

# COMMAND ----------

# DBTITLE 1,Draw boxplot of comparison
pred_df_savings_pd_clip = pred_df_savings_pd.copy()

with sns.axes_style("whitegrid"):
    g = sns.catplot(
        data=pred_df_savings_pd_clip,
        x="Baseline Fuel",
        y="Absolute Error (kWh)",
        order=[
            "Fuel Oil",
            "Propane",
            "Methane Gas",
            "Electricity",
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
        sharey=False,
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
g.fig.suptitle("Model Prediction Comparison: Total Annual Energy Savings")

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
    save_figure_to_gcfs(
        g.fig,
        EXPORT_FPATH
        / "surrogate_model_metrics"
        / "comparison"
        / f"{MODEL_RUN_NAME}_vs_bucketed.png",
    )

# COMMAND ----------


