from pyspark.sql import SparkSession

def get_dbutils(spark):
        try:
            from pyspark.dbutils import DBUtils
            dbutils = DBUtils(spark)
        except ImportError:
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
        return dbutils

# -- databricks vars that are not native in .py files -- #      
spark =  SparkSession.builder.getOrCreate()
dbutils = get_dbutils(spark)
