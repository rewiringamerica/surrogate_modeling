# Databricks notebook source
# MAGIC %md # Test to make sure model works in different notebook
# MAGIC

# COMMAND ----------

# install required packages: note that tensorflow must be installed at the notebook-level
%pip install tensorflow==2.15.0.post1

# COMMAND ----------

from src.databricks.model import Model

# COMMAND ----------

# init a model
model = Model(name="test")

# COMMAND ----------

# #pull in some inference data with the right shape
test_data = spark.sql(
    "SELECT building_id, upgrade_id, weather_file_city FROM ml.surrogate_model.building_features LIMIT 10"
)

# COMMAND ----------

# test predictions and make sure nothing errors out
import mlflow

mlflow.pyfunc.get_model_dependencies(model.get_model_uri())

pred_df = model.score_batch(test_data=test_data)  # score using  latest registered model

# pred_df = model.score_batch(test_data = test_data, run_id = '6deab2c9e96a402ab0bf2c6d1108f53e') # score unregistered model

pred_df.display()
#pred_df = model.score_batch(test_data = test_data, run_id = '6deab2c9e96a402ab0bf2c6d1108f53e', targets = ['heating', 'cooling']) # score unregistered model

pred_df.display()

# COMMAND ---------
