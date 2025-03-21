# Databricks notebook source
# MAGIC %md # Compare Performance of Surrogate Model Versions
# MAGIC
# MAGIC Compares the evaluation metrics between two different surrogate models based on previously written out predictions on a held out test set. Write out csvs and figures to gcs and local artifacts.

# COMMAND ----------

# MAGIC %pip install seaborn==v0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widget setup
dbutils.widgets.text("sumo_version_num_previous", "01_00_00")
dbutils.widgets.text("sumo_version_num_new", "01_01_00")

# COMMAND ----------

# DBTITLE 1,Import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as F

from dmlutils.building_upgrades import upgrades
from dmlutils.gcs import save_fig_to_gcs

from src.globals import GCS_ARTIFACT_PATH, LOCAL_ARTIFACT_PATH

pd.set_option('display.max_rows', 100) 

# COMMAND ----------

version_num_prev = dbutils.widgets.get("sumo_version_num_previous")
version_num_new = dbutils.widgets.get("sumo_version_num_new")

# COMMAND ----------

# MAGIC %md ## Load test set error metrics and aggregated error metrics

# COMMAND ----------

predictions_version_prev = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_prev / "prediction_metrics_test_set.csv")).replace({"Methane Gas": "Natural Gas"})
predictions_version_new = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_new / "prediction_metrics_test_set.csv"))

# COMMAND ----------

aggregated_metrics_prev = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_prev / "metrics_by_upgrade_type.csv")).replace({"Methane Gas": "Natural Gas"})
aggregated_metrics_new = pd.read_csv(str(GCS_ARTIFACT_PATH / version_num_new  / "metrics_by_upgrade_type.csv"))

# COMMAND ----------

aggregated_metrics_new.to_csv(str(LOCAL_ARTIFACT_PATH  / "metrics_by_upgrade_type.csv"), index=False)

# COMMAND ----------

# MAGIC %md ## Compute table of aggregated diffs

# COMMAND ----------

def compute_diffs(df1, df2, index_cols):
    """
    Compute the differences between two DataFrames on non-key columns.

    Parameters:
    ----------
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        index_cols (list): List of columns to set as index for alignment.

    Returns:
    -------
        pd.DataFrame: DataFrame containing the differences between df2 and df1.
    """
    # Set index for alignment
    df1 = df1.set_index(index_cols)
    df2 = df2.set_index(index_cols)

    # Ensure both DataFrames have the same structure
    if not df1.columns.equals(df2.columns):
        raise ValueError("DataFrames must have the same non-index columns")

    # Compute the difference
    df_diff = df2 - df1

    # Reset index to return the original format
    return df_diff

# COMMAND ----------

#compute table of the differences in error for each (upgrade, type), where positive numbers indicate higher error for the new version
df_diffs = compute_diffs(
  aggregated_metrics_prev,
  aggregated_metrics_new,
  index_cols=['Upgrade ID', 'Type'])
df_diffs.to_csv(str(GCS_ARTIFACT_PATH / "metrics_change_from_previous_version_by_upgrade_type.csv"))
df_diffs.to_csv(str(LOCAL_ARTIFACT_PATH / "metrics_change_from_previous_version_by_upgrade_type.csv"))

# COMMAND ----------

# MAGIC %md ### Visualize

# COMMAND ----------

#create mapping of upgrade id -> upgrade name
upgrade_name_df = upgrades.upgrades_df(spark).select(F.col('name').alias('Upgrade Name'), 'upgrade_id').toPandas()
upgrade_name_df['upgrade_id'] = upgrade_name_df['upgrade_id'].astype(float)

# COMMAND ----------

# combine both versions into one dataframe
predictions_combined_versions = pd.concat([
  predictions_version_prev.assign(Version=version_num_prev),
  predictions_version_new.assign(Version=version_num_new)], ignore_index=True)

# COMMAND ----------

# Process for plotting:
# * subset to total over all fuels
# set metrics of interest to savings for non-baseline upgrades and absolute for baseline upgrades
# remove baseline types that have very high error that we don't support
# rename to cleaner labels

predictions_combined_versions_savings_total = (
    predictions_combined_versions
        .query("fuel == 'total'")
        .query("baseline_appliance not in ['Heat Pump', 'No Heating']")
        .assign(
            absolute_error=lambda df: df["absolute_error"]
            .where(df["upgrade_id"].isin([0, 0.01]), df["absolute_error_savings"])
        )
        .replace({"Electric Resistance": "Electricity"})
        .merge(upgrade_name_df, on = 'upgrade_id')
        .rename(
            columns={
                "baseline_appliance": "Baseline Fuel",
                "absolute_error": "Absolute Error (kWh)",
                "upgrade_id": "Upgrade ID"
            }
        )
)

# Combine upgreade id and name
predictions_combined_versions_savings_total['Upgrade Name'] = predictions_combined_versions_savings_total['Upgrade Name'] + ' (' + predictions_combined_versions_savings_total['Upgrade ID'].astype(str) + ')'

# COMMAND ----------

def plot_error_comparison_boxplot(data, x, y="Absolute Error (kWh)", row=None, order = None, title = None):
    """
    Plots a comparison boxplot of prediction errors.

    Parameters:
    -----------
        data (DataFrame): The data to plot.
        x (str): The column name to be used for the x-axis.
        y (str, optional): The column name to be used for the y-axis. Default is "Absolute Error (kWh)".
        row (str, optional): The column name to facet subplots by rowwise. Default is None.
        order (list, optional): The order of categories for the x-axis. Default is None.
        title (str, optional): The title of the plot. Default is None.

    Returns:
    --------
    Figure: The resulting matplotlib figure.
    """
    with sns.axes_style("whitegrid"):
        g = sns.catplot(
            data=data,
            x=x,
            y=y,
            hue="Version",
            order=order,
            palette="viridis",
            fill=False,
            linewidth=1.25,
            kind="box",
            row=row,
            sharey=False,
            sharex=True,
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
    if title:
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
    return g.fig

# COMMAND ----------

# Plot absolute error by upgrade and type
fig = plot_error_comparison_boxplot(
    data = predictions_combined_versions_savings_total.sort_values("Upgrade ID"),
    x="Baseline Fuel",
    order=[
        "Fuel Oil",
        "Propane",
        "Natural Gas",
        "Electricity"
    ],
    row="Upgrade Name",
    title="Model Prediction Comparison by Upgrade and Baseline Fuel Type: Total Annual Energy Savings")
fig.savefig(LOCAL_ARTIFACT_PATH / "model_prediction_comparison_boxplot_by_upgrade_type.png")
save_fig_to_gcs(fig, GCS_ARTIFACT_PATH / version_num_new /  "model_prediction_comparison_boxplot_by_upgrade_type.png")

# COMMAND ----------

# Plot absolute error by upgrade
fig = plot_error_comparison_boxplot(
    data = predictions_combined_versions_savings_total,
    x = "Upgrade ID",
    title="Model Prediction Comparison by Upgrade: Total Annual Energy Savings")
# add a dotted line between baseline and other upgrades since this is absolute error rather than savings error plotted
plt.axvline(x=1.5, color='gray', linestyle='--', linewidth=1.5)

fig.savefig(LOCAL_ARTIFACT_PATH / "model_prediction_comparison_boxplot_by_upgrade.png")
save_fig_to_gcs(fig, GCS_ARTIFACT_PATH / version_num_new /  "model_prediction_comparison_boxplot_by_upgrade.png")

