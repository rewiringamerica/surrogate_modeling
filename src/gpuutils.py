"""Utilities for measuring GPU performance."""

from typing import Optional

from logging import getLogger

from pynvml import *
import psutil

import pandas as pd

import time


logger = getLogger(__file__)


def gather_cpu_gpu_metrics(
    interval: int | pd.Timedelta = 5,
    iterations: int = 12,
    *,
    gather_for_time: Optional[pd.Timedelta] = None,
    log_interval: Optional[int] = None,
):
    """Gather GPU performance at regular intervals."""

    if not isinstance(interval, pd.Timedelta):
        interval = pd.Timedelta(interval, unit="seconds")

    try:
        nvmlInit()

        handle = nvmlDeviceGetHandleByIndex(0)

        start_time = pd.Timestamp.now()

        raw_data = []

        if gather_for_time is None:
            if iterations <= 0:
                return
            end_time = None
        else:
            end_time = pd.Timestamp.now() + gather_for_time

        intervals_run = 0

        while True:
            iteration_start_time = pd.Timestamp.now()

            utilization = nvmlDeviceGetUtilizationRates(handle)

            gpu_utilization = utilization.gpu
            gpu_memory_utilization = utilization.memory

            cpu_utilization = psutil.cpu_percent()
            ram_utilization = psutil.virtual_memory().percent

            yield (
                {
                    "timestamp": iteration_start_time,
                    "cpu_utilization": cpu_utilization,
                    "ram_utilization": ram_utilization,
                    "gpu_utilization": gpu_utilization,
                    "gpu_memory_utilization": gpu_memory_utilization,
                }
            )

            if end_time is not None:
                if pd.Timestamp.now() >= end_time:
                    break
            else:
                if intervals_run >= iterations:
                    break

            next_time = start_time + interval * (intervals_run + 1)
            sleep_duration = next_time - pd.Timestamp.now()
            time.sleep(sleep_duration.total_seconds())

            intervals_run += 1

            if log_interval is not None and intervals_run % log_interval == 0:
                logger.info(
                    f"{intervals_run} intervals run in {pd.Timestamp.now() - start_time}."
                )
    finally:
        nvmlShutdown()
