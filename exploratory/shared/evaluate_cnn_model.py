# Databricks notebook source
# MAGIC %md # Evaluate CNN Model
# MAGIC

# COMMAND ----------

# MAGIC %pip install mlflow==2.13.0 seaborn==v0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import mlflow
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import pyspark.sql.functions as F
# MAGIC import seaborn as sns
# MAGIC from functools import reduce
# MAGIC from pyspark.sql import DataFrame
# MAGIC from pyspark.sql.window import Window
# MAGIC from pyspark.sql.types import ArrayType, IntegerType
# MAGIC
# MAGIC from src.databricks.datagen import DataGenerator, load_data
# MAGIC from src.databricks.surrogate_model import SurrogateModel

# COMMAND ----------

MODEL_NAME = "sf_hvac_by_fuel"
RUN_ID = "21170c4978fd4efc9d90e25d70880d76"

# COMMAND ----------

sm = SurrogateModel(name=MODEL_NAME)
_, _, test_data = load_data(n_test=100000)
test_gen = DataGenerator(test_data)
test_set = test_gen.init_training_set(
    test_data, exclude_columns=["weather_file_city"]
).load_df()

# COMMAND ----------

mlflow.pyfunc.get_model_dependencies(model_uri=sm.get_model_uri(run_id=RUN_ID))
# Load the model using its registered name and version/stage from the MLflow model registry
model_loaded = mlflow.pyfunc.load_model(model_uri=sm.get_model_uri(run_id=RUN_ID))
# load input data table as a Spark DataFrame
inference_data = test_set.toPandas()

# COMMAND ----------

# run inference: takes ~20s on 10,0000 and ~2m on 100,0000 samples
prediction_arr = model_loaded.predict(inference_data)

# COMMAND ----------

targets_formatted = np.array(
    [item.replace("_", " ").title() for item in test_gen.targets]
)
fuel_present_by_sample_appliance_fuel = np.expand_dims(
    targets_formatted, axis=[0, 1]
) == np.expand_dims(inference_data[["heating_fuel"]], 2)
fuel_present_by_sample_fuel_mask = fuel_present_by_sample_appliance_fuel.sum(1).astype(
    bool
)
fuel_present_by_sample_fuel_mask[
    :, targets_formatted == "Electricity"
] = True  # all(ish) homes have electricity so set this to always be 1
# null out the predictions if there are no appliances with that fuel type
predictions_with_nulled_out_fuels = np.where(
    ~fuel_present_by_sample_fuel_mask, np.nan, prediction_arr
)

# COMMAND ----------

targets = test_gen.targets + ["total"]

sample_pkeys = ["building_id", "upgrade_id"]
sample_pkey_arr = inference_data[sample_pkeys].values
predictions_with_nulled_out_fuels_and_building_ids = np.hstack(
    [
        sample_pkey_arr,  # columns of pkeys
        predictions_with_nulled_out_fuels,  # columns for each fuel
        np.expand_dims(
            np.nansum(predictions_with_nulled_out_fuels, 1), 1
        ),  # add column for totals summed over all fuels
    ]
)

# COMMAND ----------

# Create a N x M DataFrame where N is the nummber of test samples and M is the number of targets
target_pred_labels = [f"{y}-pred" for y in targets]
pred_only = spark.createDataFrame(
    predictions_with_nulled_out_fuels_and_building_ids.tolist(),
    sample_pkeys + target_pred_labels,
)

# Create dataframe with columns for actual and predicted values for each fuel
# and add a total columns to the actual
pred_wide = pred_only.join(
    test_data.withColumn("total", F.expr("+".join(test_gen.targets))), on=sample_pkeys
)

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

# add metadata
keep_features = ["heating_fuel", "heating_appliance_type", "ac_type"]
pred_by_building_upgrade_fuel = test_set.select(*sample_pkeys, *keep_features).join(
    pred_by_building_upgrade_fuel, on=sample_pkeys
)

# COMMAND ----------

def aggregate_metrics(pred_df_savings, groupby_cols, target_idx):
 
    aggregation_expression = [
        F.round(f(F.col(colname)[target_idx]), round_precision).alias(f"{f.__name__}_{colname}")
        for f in [F.median, F.mean]
        for colname, round_precision in [
            ("absolute_error", 0),
            ("absolute_percentage_error", 1),
            ("absolute_error_savings", 0),
            ("absolute_percentage_error_savings", 1)
        ]
    ]

    return (
        pred_df_savings
            .groupby(*groupby_cols)
            .agg(*aggregation_expression)
        )

# COMMAND ----------
cooling_metrics_by_type_upgrade = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(F.col("baseline_ac_type") != "Heat Pump")
        .where(F.col("fuel") == "total")
        .where(F.col("upgrade_id") == 0),
        groupby_cols=["baseline_ac_type", "upgrade_id"],
    ).withColumnRenamed("baseline_ac_type", "type")
)

heating_metrics_by_type_upgrade = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
        groupby_cols=["baseline_heating_fuel", "upgrade_id"],
    ).withColumnRenamed("baseline_heating_fuel", "type")
)

total_metrics_by_upgrade = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(F.col("fuel") == "total"),
        groupby_cols=["upgrade_id"],
    )
    .withColumn('type', F.lit('Total'))

total_metrics_by_upgrade_fuel = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(F.col("prediction").isNotNull()).where(
            F.col("fuel") != "total"
        ),
        groupby_cols=["fuel", "upgrade_id"],
    ).withColumn("type", F.concat_ws(" : ", F.lit("Total"), F.col("fuel")))
    .drop("fuel")
)

dfs = [
    heating_metrics_by_type_upgrade,
    cooling_metrics_by_type_upgrade,
    total_metrics_by_upgrade,
    total_metrics_by_upgrade_fuel,
]
cnn_evaluation_metrics = reduce(DataFrame.unionByName, dfs)
cnn_evaluation_metrics_without_fuel = reduce(DataFrame.unionByName, dfs[:-1])

# COMMAND ----------

# save the metrics table tagged with the model name and version number
(
    cnn_evaluation_metrics
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("userMetadata", {"run_id": RUN_ID, "model_name": MODEL_NAME})
        .saveAsTable("ml.surrogate_model.evaluation_metrics")
)

# COMMAND ----------

# MAGIC %md ## Export Metrics

# COMMAND ----------

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
# set the order in which types will appear in the table
types = [
    "Electric Resistance",
    "Natural Gas",
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
    "Total : natural_gas",
    "Total : fuel_oil",
    "Total : propane",
    "Total : electricity",
]

# COMMAND ----------



# COMMAND ----------
# MAGIC %md ## Compare against Bucketed Model

# COMMAND ----------

# bring in bucketed metrics and compare against that.
bucket_metrics = pd.read_csv(
    "gs://the-cube/export/surrogate_model_metrics/bucketed_sf_hvac.csv",
    keep_default_na=False,
    dtype={"upgrade_id": "double"},
)
cnn_metrics = cnn_evaluation_metrics_without_fuel.toPandas()

# COMMAND ----------

bucket_metrics["Model"] = "Bucketed"
cnn_metrics["Model"] = "CNN"

# COMMAND ----------

metrics_combined = (
    pd.concat([cnn_metrics, bucket_metrics])
    .rename(
        columns={**metric_rename_dict, **{"upgrade_id": "Upgrade ID", "type": "Type"}}
    )
    .melt(
        id_vars=["Type", "Model", "Upgrade ID"],
        value_vars=list(metric_rename_dict.values()),
        var_name="Metric",
    )
)

metrics_combined["Type"] = pd.Categorical(
    metrics_combined["Type"], categories=types, ordered=True
)

metrics_combined = metrics_combined.pivot(
    index=[
        "Upgrade ID",
        "Type",
    ],
    columns=["Metric", "Model"],
    values="value",
)

# COMMAND ----------

metrics_combined 

# COMMAND ----------

metrics_combined.to_csv(
    f"gs://the-cube/export/surrogate_model_metrics/comparison/{MODEL_NAME}_by_method_upgrade_type.csv",
    float_format="%.2f",
)

# COMMAND ----------

# MAGIC %md ## Visualize Comparison between Model Metrics

# COMMAND ----------

pred_df_savings_hvac = pred_df_savings.withColumn(
    "absolute_percentage_error",
    F.when(F.col("upgrade_id") == 0, F.col("absolute_percentage_error")[4]).otherwise(
        F.col("absolute_percentage_error_savings")[4]
    ),
).select("upgrade_id", "baseline_heating_fuel", "absolute_percentage_error")

# COMMAND ----------

bucketed_pred = (
    spark.table("ml.surrogate_model.bucketed_sf_hvac_predictions")
    .withColumn(
        "absolute_percentage_error",
        F.when(F.col("upgrade_id") == 0, F.col("absolute_percentage_error")).otherwise(
            F.col("absolute_percentage_error_savings")
        ),
    )
    .select(
        "upgrade_id",
        F.col("baseline_appliance_fuel").alias("baseline_heating_fuel"),
        "absolute_percentage_error",
    )
)

# COMMAND ----------

pred_df_savings_pd = (
    pred_df_savings_hvac
        .withColumn('Model', F.lit('CNN'))
        .unionByName(bucketed_pred.withColumn('Model', F.lit('Bucketed')))
        .replace({'Shared Heating' : 'Shared', 'No Heating' : 'None'}, subset = 'baseline_heating_fuel')
        .where(F.col('baseline_heating_fuel') != 'Heat Pump')
        .withColumnsRenamed(
            {'baseline_heating_fuel' : 'Baseline Heating Fuel', 
             "absolute_percentage_error" : "Absolute Percentage Error", 
             'upgrade_id' : 'Upgrade ID'})
    ).toPandas()

# COMMAND ----------

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
        row_order=["0", "1", "3", "4"],
        height=3,
        aspect=3,
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

# TODO: move to a utils file once code is reorged
import io
from google.cloud import storage
from cloudpathlib import CloudPath


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

save_figure_to_gcfs(
    g.fig,
    CloudPath("gs://the-cube")
    / "export"
    / "surrogate_model_metrics"
    / "comparison"
    / f"{MODEL_VERSION_NAME}_vs_bucketed.png",
)

# COMMAND ----------

