# Databricks notebook source
# MAGIC %md # Test to make sure model works in different notebook
# MAGIC

# COMMAND ----------

# install required packages: note that tensorflow must be installed at the notebook-level
%pip install gcsfs==2023.5.0 tensorflow==2.15.0.post1

# COMMAND ----------

from src.databricks.model import Model

# COMMAND ----------

#init a model
model = Model(name='test')

# COMMAND ----------


from src.databricks.datagen import DataGenerator

consumption_group_dict= DataGenerator.consumption_group_dict
building_feature_table_name= DataGenerator.building_feature_table_name

# join to the bm table to get required keys to join on and filter the building models based on charactaristics 
sum_str = ", ".join(
    [f"{'+'.join(v)} AS {k}" for k, v in consumption_group_dict.items()]
)


test_data = spark.sql(
    f"""
    SELECT B.building_id, B.upgrade_id, B.weather_file_city, {sum_str}
    FROM ml.surrogate_model.building_upgrade_simulation_outputs_annual O
    LEFT JOIN {building_feature_table_name} B 
        ON B.upgrade_id = O.upgrade_id AND B.building_id == O.building_id
    WHERE sqft < 8000
        AND occupants <= 10
        AND O.building_id < 100
    """
)

# COMMAND ----------

# #pull in some inference data with the right shape
test_data = spark.sql("SELECT building_id, upgrade_id, weather_file_city FROM ml.surrogate_model.building_features LIMIT 10")

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
from pyspark.sql.types import ArrayType, DoubleType

# COMMAND ----------

# test predictions and make sure nothing errors out
import mlflow
model_uri = "models:/ml.surrogate_model.test/36"
mlflow.pyfunc.get_model_dependencies(model_uri)


batch_pred = fe.score_batch(
    model_uri=model_uri,
    df=test_data,
    result_type=ArrayType(DoubleType())
)

# COMMAND ----------


batch_pred.display()

# COMMAND ----------

# test predictions and make sure nothing errors out
import mlflow
mlflow.pyfunc.get_model_dependencies(model.get_model_uri())

pred_df = model.score_batch(test_data = test_data) # score using  latest registered model

#pred_df = model.score_batch(test_data = test_data, run_id = '6deab2c9e96a402ab0bf2c6d1108f53e', targets = ['heating', 'cooling']) # score unregistered model

# COMMAND ----------

pred_df.display()

# COMMAND ----------


