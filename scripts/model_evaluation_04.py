# Databricks notebook source
# MAGIC %md # Evaluate Surrogate Model
# MAGIC
# MAGIC ### Goal
# MAGIC Predict on a held out test set and then summarize aggregated metrics. 
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC The passed in mlflow `run_id` determines which saved surrogate model is evaluated. The `test_size` determines the number of samples to perform inference on (subset from the pre-defined held out test set). *If running on a large test set (>10,000), a GPU is recommended*.
# MAGIC
# MAGIC ##### Outputs:
# MAGIC Outputs are written based on the current version number of this repo in `pyproject.toml`.
# MAGIC         str(GCS_CURRENT_VERSION_ARTIFACT_PATH / "prediction_metrics_test_set.csv"),
# MAGIC - `gs://the-cube/export/surrogate_model/model_artifacts/{CURRENT_VERSION_NUM}/prediction_metrics_test_set.csv`: predictions, actuals and errors for each (building_id, upgrade_id, fuel), also tagged with important metadata such as baseline appliance fuel and cooling type. 
# MAGIC - `gs://the-cube/export/surrogate_model/model_artifacts/{CURRENT_VERSION_NUM}/metrics_by_upgrade_type.csv`: aggregated errors for each (upgrade_id, type) where type categories vary based on the upgrade. 

# COMMAND ----------

# MAGIC %pip install mlflow==2.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget setup
dbutils.widgets.dropdown("mode", "test", ["test", "production"])
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("test_size", "10000")  # default in test mode

# COMMAND ----------

DEBUG = dbutils.widgets.get("mode") == "test"
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
# MAGIC from pyspark.sql import DataFrame, Column
# MAGIC from pyspark.sql.window import Window
# MAGIC
# MAGIC # NOTE: for now this cannot depend on dmlutils because training/testing requires a GPU cluster
# MAGIC # and we do not yet have a requirements file that is compatible for both dmlutils and the GPU cluster
# MAGIC # from dmlutils.gcs import save_fig_to_gcs
# MAGIC
# MAGIC from src.globals import GCS_CURRENT_VERSION_ARTIFACT_PATH
# MAGIC from src.datagen import DataGenerator, load_data
# MAGIC from src.surrogate_model import SurrogateModel

# COMMAND ----------

# MAGIC %md ## Evaluate Model

# COMMAND ----------

# MAGIC %md ### Load Model and Inference Data

# COMMAND ----------

# DBTITLE 1,Load model
# init model
sm = SurrogateModel()
# mlflow.pyfunc.get_model_dependencies(model_uri=sm.get_model_uri(run_id=RUN_ID))
# Load the unregistered model using run ID
model_loaded = mlflow.pyfunc.load_model(model_uri=sm.get_model_uri(run_id=RUN_ID))

# COMMAND ----------

# DBTITLE 1,Load test data
# load test data
_, _, test_data = load_data(n_test=TEST_SIZE)
#init data generator so we can get all of the features -- note that we can't use databricks automatic lookup of features since we logged with mlflow
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
# takes ~20s on 10,000 and ~2m on 100,0000 samples
prediction_arr = model_loaded.predict(inference_data)

# COMMAND ----------

# DBTITLE 1,Concatenate pkeys and predictions
# create an array of N x 2 with the pkeys needed to join this to metadata
sample_pkeys = ["building_id", "upgrade_id", "building_set"]
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

# MAGIC %md ## Combine actual, predictions and calculate savings

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
w_building = Window().partitionBy("building_id", "fuel", "building_set").orderBy(F.asc("upgrade_id"))

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
)

# COMMAND ----------

# DBTITLE 1,Add and transform metadata
# add metadata that we will want to cut up results by
baseline_appliance_features = [
    "heating_fuel",
    "ac_type",
    "water_heater_fuel",
]

pred_by_building_upgrade_fuel_model_with_metadata = test_set.select(
    *sample_pkeys, *baseline_appliance_features
).join(pred_by_building_upgrade_fuel_savings, on=sample_pkeys)

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
# 2. Calculate absolute error
# 3. Calculate savings absolute error, setting these to Null for baseline since there is no savings
# 4. Calculate APE for consumption and savings
pred_df_savings = (
    pred_by_building_upgrade_fuel_model_with_metadata.withColumn(  # 1
        "baseline_appliance",
        F.when(
            F.col("upgrade_id") == 6,
            F.first(F.col("water_heater_fuel")).over(w_building),
        )
        .otherwise(F.first(F.col("heating_fuel")).over(w_building)),
    )
    .withColumn("baseline_ac_type", F.first(F.col("ac_type")).over(w_building))
    .withColumn(  # 2
        "absolute_error",F.abs(F.col("prediction") - F.col("actual"))
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
).drop(*baseline_appliance_features)

# COMMAND ----------

# write out metrics for each sample in the test set to gcs so that analyses can be performed subsequently
if not DEBUG:
    pred_df_savings.toPandas().to_csv(
        str(GCS_CURRENT_VERSION_ARTIFACT_PATH / "prediction_metrics_test_set.csv"),
        index=False
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
    groupby_cols=["baseline_ac_type", "upgrade_id"],
).withColumnRenamed("baseline_ac_type", "type")

# calculate metrics by upgrade and baseline heating fuel
heating_metrics_by_type_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
    groupby_cols=["baseline_appliance", "upgrade_id"],
).withColumnRenamed("baseline_appliance", "type")

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
metrics_by_upgrade_type = reduce(DataFrame.unionByName, dfs)

# COMMAND ----------

metrics_by_upgrade_type.display()

# COMMAND ----------

# MAGIC %md ## Format for export

# COMMAND ----------

# DBTITLE 1,Convert to pandas
metrics_by_upgrade_type_pd = metrics_by_upgrade_type.toPandas()

# COMMAND ----------

# DBTITLE 1,Rename columns for readability
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

key_rename_dict = {"upgrade_id": "Upgrade ID", "type": "Type"}

metrics_by_upgrade_type_pd.rename(
    columns={**metric_rename_dict, **key_rename_dict}, inplace=True
)

# COMMAND ----------

# DBTITLE 1,Specify row order
# set the order in which types will appear in the table
metrics_by_upgrade_type_pd["Type"] = pd.Categorical(
    metrics_by_upgrade_type_pd["Type"],
    categories=[
        "Electricity",
        "Electric Resistance",
        "Natural Gas",
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

metrics_by_upgrade_type = metrics_by_upgrade_type_pd.sort_values(["Upgrade ID", "Type"])

# COMMAND ----------

# DBTITLE 1,Write results to csv
if not DEBUG:
    # write to gcs-- can't write locally in a job so can copy it to the repo later if desired
    metrics_by_upgrade_type.to_csv(
        str(GCS_CURRENT_VERSION_ARTIFACT_PATH / "metrics_by_upgrade_type.csv"),
        index=False
    )
