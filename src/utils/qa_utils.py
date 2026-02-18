# TODO: move to dmlutils

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def check_for_null_values(df: DataFrame) -> None:
    """
    Check for null values in each column of the DataFrame and raises an error if any null values are found.

    Args
    ----
    df (DataFrame): The DataFrame to check for null values.

    Raises
    ------
    ValueError: If any column contains null values, an error is raised listing those columns and their null counts.
    """
    # count how many null vals are in each column
    null_counts = df.select([F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns])
    # Collect the results as a dictionary
    null_counts_dict = null_counts.collect()[0].asDict()
    # select any items that have non-zero count
    null_count_items = {k: v for k, v in null_counts_dict.items() if v > 0}
    if len(null_count_items) > 0:
        raise ValueError(f"Columns with null values: {null_count_items}")


def compare_dataframes_string_values(df1: DataFrame, df2: DataFrame) -> dict:
    """
    Identify the differences possible values for string columns in two PySpark DataFrames.

    Args:
        df1 (DataFrame): The first dataframe to compare.
        df1 (DataFrame): The second dataframe to compare.

    Returns
    -------
        dict: A dictionary of differences between the two DataFrames. The keys are the string columns that
        have differences, and the values are dictionaries containing the differences for each column.
    """
    # Initialize a dictionary to store the results
    comparison_dict = {}

    # Get string columns
    string_cols_df1 = [field.name for field in df1.schema.fields if field.dataType.simpleString() == "string"]
    string_cols_df2 = [field.name for field in df2.schema.fields if field.dataType.simpleString() == "string"]

    # Find common string columns
    common_string_cols = set(string_cols_df1).intersection(set(string_cols_df2))

    for col in common_string_cols:
        # Get unique values as lists
        unique_df1 = df1.select(col).distinct().rdd.flatMap(lambda x: x).collect()
        unique_df2 = df2.select(col).distinct().rdd.flatMap(lambda x: x).collect()

        unique_set1 = set(unique_df1)
        unique_set2 = set(unique_df2)

        differences = {}

        # Find values unique to df1
        only_in_df1 = unique_set1 - unique_set2
        if only_in_df1:
            differences["df1 only"] = only_in_df1

        # Find values unique to df2
        only_in_df2 = unique_set2 - unique_set1
        if only_in_df2:
            differences["df2 only"] = only_in_df2

        if differences:
            comparison_dict[col] = differences

    return comparison_dict
