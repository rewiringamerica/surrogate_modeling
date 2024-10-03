from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def check_for_null_values(df: DataFrame) -> None:
    """
    Checks for null values in each column of the DataFrame and raises an error if any null values are found.

    Args:
    df (DataFrame): The DataFrame to check for null values.

    Raises:
    ValueError: If any column contains null values, an error is raised listing those columns and their null counts.
    """
    # count how many null vals are in each column
    null_counts = df.select(
        [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ]
    )
    # Collect the results as a dictionary
    null_counts_dict = null_counts.collect()[0].asDict()
    # select any items that have non-zero count
    null_count_items = {k: v for k, v in null_counts_dict.items() if v > 0}
    if len(null_count_items) > 0:
        raise ValueError(f"Columns with null values: {null_count_items}")