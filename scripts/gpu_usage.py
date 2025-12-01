# Databricks notebook source
# DBTITLE 1,Install python bindings for NVIDIA Management Library
# MAGIC %pip install nvidia-ml-py

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

# TODO: Probably want to move this to dmlutils.
import src.utils.gpuutils as gpuutils

# COMMAND ----------

import logging

logging.basicConfig(level=logging.INFO)

# COMMAND ----------

df_usage = pd.DataFrame()

# COMMAND ----------

# This does not seem to interactively update the plot for each
# new iteration in databricks. I'm not sure what has to be done
# differently than in a normal Jupyter notebook.

# But the code is set up so you can interrupt this cell, then
# use the next two cells to look at the currently data that has
# been collected and plot it.

plt.ion()

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("CPU, RAM, GPU and GPU Memory Utilization")

label_map = {
    "cpu_utilization": "CPU",
    "ram_utilization": "RAM",
    "gpu_utilization": "GPU",
    "gpu_memory_utilization": "GPU Memory",
}

lines = None

for row in gpuutils.gather_cpu_gpu_metrics(
    interval=pd.Timedelta(2, unit="seconds"),
    gather_for_time=pd.Timedelta(30, unit="minutes"),
    log_interval=10,
):
    df_usage = pd.concat([df_usage, pd.DataFrame([row])])

    data_cols = [col for col in df_usage.columns if col != "timestamp"]
    labels = [label_map[col] for col in data_cols]

    if lines is None:
        lines = ax.plot(
            df_usage["timestamp"],
            df_usage[data_cols],
            label=labels,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Utilization")

        ax.grid()
    else:
        for line, col in zip(
            lines,
            data_cols,
        ):
            line.set_xdata(df_usage["timestamp"])
            line.set_ydata(df_usage[col])

    ax.legend()
    ax.relim()
    ax.set_ylim(0.0, 100.0)
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# COMMAND ----------

df_usage

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title(", ".join(labels) + " Utilization")

lines = ax.plot(
    df_usage["timestamp"],
    df_usage[data_cols],
    label=labels,
)
ax.set_xlabel("Time")
ax.set_ylabel("Utilization")

ax.set_ylim(0.0, 100.0)

ax.legend()
ax.grid()
