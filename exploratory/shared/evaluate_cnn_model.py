# Databricks notebook source
# MAGIC %md # Evaluate CNN Model
# MAGIC

# COMMAND ----------

!pip install seaborn==v0.13.0
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from pyspark.sql.window import Window

# COMMAND ----------

MODEL_NAME = "sf_detatched_hvac"
MODEL_TESTSET_PREDICTIONS_TABLE = f"ml.surrogate_model.{MODEL_NAME}_predictions"
MODEL_VERSION_NUMBER = (
    spark.sql(
        f"SELECT userMetadata FROM (DESCRIBE HISTORY {MODEL_TESTSET_PREDICTIONS_TABLE }) ORDER BY version DESC LIMIT 1"
    )
    .rdd.map(lambda x: x["userMetadata"])
    .collect()[0]
)
MODEL_VERSION_NAME = f"ml.surrogate_model.{MODEL_NAME}@v{MODEL_VERSION_NUMBER}"

# COMMAND ----------

targets = [
    "heating",
    "cooling",
]  # in theory could prob pull this from model artifacts..
pred_df = spark.table(MODEL_TESTSET_PREDICTIONS_TABLE)
building_features = spark.table("ml.surrogate_model.building_features")

pred_df = pred_df.drop("heating_fuel", "ac_type").join(
    building_features, on=["building_id", "upgrade_id"]
)

# COMMAND ----------

@udf("array<double>")
def APE(prediction, actual, eps=1e-3):
    return [
        abs(float(x - y)) / y * 100 if abs(y) > eps else None
        for x, y in zip(prediction, actual)
    ]

# COMMAND ----------

w = Window().partitionBy("building_id").orderBy(F.asc("upgrade_id"))


def element_wise_subtract(a, b):
    return F.expr(f"transform(arrays_zip({a}, {b}), x -> abs(x.{a} - x.{b}))")


pred_df_hvac_process = (
    pred_df.replace("None", "No Heating", subset="heating_fuel")
    .replace("None", "No Cooling", subset="ac_type")
    .withColumn(
        "heating_fuel",
        F.when(F.col("ac_type") == "Heat Pump", F.lit("Heat Pump"))
        .when(F.col("heating_fuel") == "Electricity", F.lit("Electric Resistance"))
        .when(F.col("heating_appliance_type") == "Shared", F.lit("Shared Heating"))
        .when(F.col("heating_fuel") == "None", F.lit("No Heating"))
        .otherwise(F.col("heating_fuel")),
    )
    .withColumn(
        "ac_type",
        F.when(F.col("ac_type") == "Shared", F.lit("Shared Cooling"))
        .when(F.col("ac_type") == "None", F.lit("No Cooling"))
        .otherwise(F.col("ac_type")),
    )
    .withColumn("hvac", F.col("heating") + F.col("cooling"))
    .withColumn("actual", F.array(targets + ["hvac"]))
    .withColumn(
        "prediction",
        F.array_insert(
            "prediction", 3, F.col("prediction")[0] + F.col("prediction")[1]
        ),
    )
)

pred_df_savings = (
    pred_df_hvac_process.withColumn(
        "baseline_heating_fuel", F.first(F.col("heating_fuel")).over(w)
    )
    .withColumn("baseline_ac_type", F.first(F.col("ac_type")).over(w))
    .withColumn("prediction_baseline", F.first(F.col("prediction")).over(w))
    .withColumn("actual_baseline", F.first(F.col("actual")).over(w))
    .withColumn(
        "prediction_savings", element_wise_subtract("prediction", "prediction_baseline")
    )
    .withColumn("actual_savings", element_wise_subtract("actual", "actual_baseline"))
    .withColumn("absolute_error", element_wise_subtract("prediction", "actual"))
    .withColumn("absolute_percentage_error", APE(F.col("prediction"), F.col("actual")))
    .withColumn(
        "absolute_error_savings",
        element_wise_subtract("prediction_savings", "actual_savings"),
    )
    .withColumn(
        "absolute_percentage_error_savings",
        APE(F.col("prediction_savings"), F.col("actual_savings")),
    )
)

# COMMAND ----------

def aggregate_metrics(pred_df_savings, groupby_cols, target_idx):

    aggregation_expression = [
        F.round(f(F.col(colname)[target_idx]), round_precision).alias(
            f"{f.__name__}_{colname}"
        )
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

heating_metrics_by_type_upgrade = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings,
        groupby_cols=["baseline_heating_fuel", "upgrade_id"],
        target_idx=2,
    )
    # target_idx=0)
    .withColumnRenamed("baseline_heating_fuel", "type")
)

# only present by cooling type for baseline
cooling_metrics_by_type_upgrade = (
    aggregate_metrics(
        pred_df_savings=pred_df_savings.where(
            F.col("baseline_ac_type") != "Heat Pump"
        ).where(F.col("upgrade_id") == 0),
        groupby_cols=["baseline_ac_type", "upgrade_id"],
        target_idx=2,
    )
    # target_idx=1)
    .withColumnRenamed("baseline_ac_type", "type")
)

total_metrics_by_upgrade = aggregate_metrics(
    pred_df_savings=pred_df_savings, groupby_cols=["upgrade_id"], target_idx=2
).withColumn("type", F.lit("Total"))

cnn_evaluation_metrics = heating_metrics_by_type_upgrade.unionByName(
    cooling_metrics_by_type_upgrade
).unionByName(total_metrics_by_upgrade)

# COMMAND ----------

cnn_evaluation_metrics.display()

# COMMAND ----------

# save the metrics table tagged with the model name and version number
(
    cnn_evaluation_metrics.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("userMetadata", MODEL_VERSION_NAME)
    .saveAsTable("ml.surrogate_model.evaluation_metrics")
)

# COMMAND ----------

# MAGIC %md ## Compare against Bucketed Model

# COMMAND ----------

# bring in bucketed metrics and compare against that.

bucket_metrics = pd.read_csv(
    "gs://the-cube/export/surrogate_model_metrics/bucketed_sf_detatched_hvac.csv",
    keep_default_na=False,
    dtype={"upgrade_id": "str"},
)
cnn_metrics = cnn_evaluation_metrics.toPandas()

# COMMAND ----------

bucket_metrics["Model"] = "Bucketed"
cnn_metrics["Model"] = "CNN"

# COMMAND ----------

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

# COMMAND ----------

metrics_combined = (
    pd.concat([bucket_metrics, cnn_metrics])
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
    metrics_combined["Type"],
    categories=[
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
    ],
    ordered=True,
)

metrics_combined = metrics_combined.pivot(
    index=[
        "Upgrade ID",
        "Type",
    ],
    columns=["Metric", "Model"],
    values="value",
).sort_values(["Upgrade ID", "Type"])

# COMMAND ----------

metrics_combined

# COMMAND ----------

metrics_combined.to_csv(
    f"gs://the-cube/export/surrogate_model_metrics/comparison/{MODEL_VERSION_NAME}_by_method_upgrade_type.csv"
)

# COMMAND ----------

# MAGIC %md ## Visualize Comparison between Model Metrics

# COMMAND ----------

pred_df_savings_hvac = pred_df_savings.withColumn(
    "absolute_percentage_error",
    F.when(F.col("upgrade_id") == 0, F.col("absolute_percentage_error")[2]).otherwise(
        F.col("absolute_percentage_error_savings")[2]
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
    pred_df_savings_hvac.withColumn("Model", F.lit("CNN"))
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

pred_df_savings_pd_clip = pred_df_savings_pd.copy()
pred_df_savings_pd_clip["Absolute Percentage Error"] = pred_df_savings_pd_clip[
    "Absolute Percentage Error"
].clip(upper=70)

with sns.axes_style("whitegrid"):

    g = sns.catplot(
        data=pred_df_savings_pd_clip,
        x="Baseline Heating Fuel",
        y="Absolute Percentage Error",
        order=["Fuel Oil", "Propane", "Natural Gas", "Electric Resistance", "None"],
        hue="Model",
        palette="viridis",
        fill=False,
        linewidth=1.25,
        kind="box",
        row="Upgrade ID",
        row_order=["0", "1", "3", "4"],
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


