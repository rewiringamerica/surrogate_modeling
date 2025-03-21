# Databricks notebook source
# MAGIC %md # Compare Performance of Surrogate Model Versions

# COMMAND ----------

# MAGIC %pip install seaborn==v0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget setup
# dbutils.widgets.text("run_id", "")

# COMMAND ----------


version_num_1 = '01_01_00'

# COMMAND ----------

# DBTITLE 1,Import
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import seaborn as sns
# MAGIC
# MAGIC from dmlutils.gcs import save_fig_to_gcs
# MAGIC
# MAGIC from src.globals import GCS_ARTIFACT_PATH
# MAGIC
# MAGIC pd.set_option('display.max_rows', 100) 
# MAGIC

# COMMAND ----------

# MAGIC %md ## Load Predictions

# COMMAND ----------

predictions_version_1 = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_1 / "prediction_metrics_test_set.csv"))

# COMMAND ----------

predictions_version_1

# COMMAND ----------

# DBTITLE 1,Write results to csv
aggregated_metrics_v1 = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_1 / "metrics_by_upgrade_type.csv"))
aggregated_metrics_v2 = pd.read_csv(str(GCS_ARTIFACT_PATH / '01_00_00' / "metrics_by_upgrade_type.csv"))

# aggregated_metrics_by_upgrade_type = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_1 / "metrics_by_upgrade_type.csv"))

# COMMAND ----------

import pandas as pd

def compute_diffs(df1, df2, index_cols):
    """Compute the differences between two DataFrames on non-key columns."""
    # Set index for alignment
    df1 = df1.set_index(index_cols)
    df2 = df2.set_index(index_cols)

    # Ensure both DataFrames have the same structure
    if not df1.columns.equals(df2.columns):
        raise ValueError("DataFrames must have the same non-index columns")

    # Compute the difference
    df_diff = df1 - df2

    # Reset index to return the original format
    return df_diff

df_diffs = compute_diffs(aggregated_metrics_v1, aggregated_metrics_v2, index_cols=['Upgrade ID', 'Type'])


# COMMAND ----------


df_diffs

# COMMAND ----------

# MAGIC %md ### Visualize

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
    )
)

# COMMAND ----------

predictions_version_1

# COMMAND ----------

# DBTITLE 1,Label cleanup
# do some cleanup to make labels more presentable, including removing baseline hps and shared heating for the sake of space

# Replace values and rename columns
predictions_version_1_total = (
    predictions_version_1.replace(
        {"No Heating": "None", "Electric Resistance": "Electricity"},
    )
    .query("baseline_appliance != 'Heat Pump'")  # Equivalent to `where`
    .rename(
        columns={
            "baseline_appliance": "Baseline Fuel",
            "absolute_error": "Absolute Error (kWh)",
            "upgrade_id": "Upgrade ID",
        }
    )
)

# do some cleanup to make labels more presentable, including removing baseline hps and shared heating for the sake of space

# Replace values and rename columns
predictions_version_2_total = (
    predictions_version_1.replace(
        {"No Heating": "None", "Electric Resistance": "Electricity"},
    )
    .query("baseline_appliance != 'Heat Pump'")  # Equivalent to `where`
    .rename(
        columns={
            "baseline_appliance": "Baseline Fuel",
            "absolute_error": "Absolute Error (kWh)",
            "upgrade_id": "Upgrade ID",
        }
    )
)


predictions_total_combined_versions = pd.concat([predictions_version_1_total.assign(Version=1), predictions_version_1_total.assign(Version=2)], ignore_index=True)

# COMMAND ----------

# DBTITLE 1,Draw boxplot of comparison
# pred_df_savings_pd_clip = pred_df_savings_pd.copy()

with sns.axes_style("whitegrid"):
    g = sns.catplot(
        data=predictions_total_combined_versions,
        x="Baseline Fuel",
        y="Absolute Error (kWh)",
        order=[
            "Fuel Oil",
            "Propane",
            "Natural Gas",
            "Electricity",
            "None",
        ],
        hue="Version",
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

with sns.axes_style("whitegrid"):
    g = sns.catplot(
        data=predictions_total_combined_versions,
        x="Upgrade ID",
        y="Absolute Error (kWh)",
        hue="Version",
        palette="viridis",
        fill=False,
        linewidth=1.25,
        kind="box",
        height=2.5,
        aspect=3.25,
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
