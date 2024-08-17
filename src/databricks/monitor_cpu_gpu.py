# Databricks notebook source
# DBTITLE 1,Install python bindings for NVIDIA Management Library
# MAGIC %pip install nvidia-ml-py

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pynvml import *
import psutil

import pandas as pd

import time

# COMMAND ----------

def gather_cpu_gpu_metrics(
    interval: int | pd.Timedelta = 5,
    iterations: int = 12
) -> pd.DataFrame:
    """Gather GPU performance at regular intervals."""

    if not isinstance(interval, pd.Timedelta):        
        interval = pd.Timedelta(interval, unit="seconds")

    nvmlInit()

    handle = nvmlDeviceGetHandleByIndex(0)

    start_time = pd.Timestamp.now()

    raw_data = []

    for ii in range(iterations):
        iteration_start_time = pd.Timestamp.now()

        utilization = nvmlDeviceGetUtilizationRates(handle)

        gpu_utilization = utilization.gpu
        gpu_memory_utilization = utilization.memory

        cpu_utilization = psutil.cpu_percent()
        ram_utilization = psutil.virtual_memory().percent

        raw_data.append(
            {
                "timestamp": iteration_start_time,
                "cpu_utilization": cpu_utilization,
                "ram_utilization": ram_utilization,
                "gpu_utilization": gpu_utilization,
                "gpu_memory_utilization": gpu_memory_utilization,
            }
        )

        if ii < iterations - 1:
            next_time = start_time + interval * (ii + 1)
            sleep_duration = next_time - pd.Timestamp.now()
            time.sleep(sleep_duration.total_seconds())

    nvmlShutdown()

    df_data = pd.DataFrame(raw_data)

    return df_data

# COMMAND ----------

df_metrics = gather_cpu_gpu_metrics(interval=1, iterations=120)

# COMMAND ----------

df_metrics

# COMMAND ----------

ax = df_metrics.plot(
    x="timestamp",
    y=[
        "cpu_utilization", "ram_utilization", 
        "gpu_utilization", "gpu_memory_utilization"
    ],
    title=f"CPU, RAM, GPU and GPU Memory Utilization (Cached Features; batch=1024)",
    figsize=(12, 8)
)

ax.set_ylim([0.0, 100.0])

ax.set_xlabel("Time")
ax.set_ylabel("Utilization")

ax.grid()

# COMMAND ----------


