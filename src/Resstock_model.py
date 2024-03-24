# Databricks notebook source
from pyspark.sql.functions import broadcast
import itertools
import math
import re
from typing import Dict
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import avg

from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
import itertools
import logging
import math
import os
import re
import calendar
from typing import Dict
import matplotlib.pyplot as plt


spark.conf.set("spark.sql.shuffle.partitions", 1536)

# COMMAND ----------


