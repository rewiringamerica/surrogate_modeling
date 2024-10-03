# TODO: Move this into dmutils

import re
from typing import List, Dict
from pyspark.sql import DataFrame

def edit_columns(
    df:DataFrame,
    remove_columns_with_substrings: List[str] = [],
    remove_substrings_from_columns: List[str] = [],
    replace_column_substrings_dict: Dict[str, str] = {},
    replace_period_character: str = "__",
) -> DataFrame:
    """
    Clean columns by doing the following in order
    1. Replace '.' with an empty string in column names so that string manipulation works
    2. Drop columns that contain strings in `remove_columns_with_strings`.
    3. Remove remove strings in `remove_strings_from_columns` from column names
    4. Replace strings that occur within column names based on `replace_strings_dict`
    It is important to note the order of operations here when constructing input arguments!

    Args:
      df (DataFrame): Input DataFrame
      remove_substrings_from_columns (list, optional): List of strings to remove from column names. Defaults to [].
      remove_columns_with_substrings (list, , optional): Remove columns that contain any of the strings in this list.
                                                         Defaults to [].
      replace_substrings_dict (dict, optional): Replace any occurances of strings within column names based on dict
                                                in format {to_replace: replace_value}.
      replace_period_character (str, optional): Character to replace '.' with. Defaults to '__'.

    Returns:
      DataFrame: Cleaned DataFrame
    """
    # replace these with an empty string
    remove_str_dict = {c: "" for c in remove_substrings_from_columns}
    # combine the two replacement lookups
    combined_replace_dict = {**replace_column_substrings_dict, **remove_str_dict}

    # Replace '.' the given character in column names so that we don't have to escape w backticks
    df = df.selectExpr(
        *[
            f" `{col}` as `{col.replace('.', replace_period_character)}`"
            for col in df.columns
        ]
    )

    # Iterate through the columns and replace dict to construct column mapping
    new_col_dict = {}
    for col in df.columns:
        # skip if in ignore list
        if len(remove_columns_with_substrings) > 0 and re.search(
            "|".join(remove_columns_with_substrings), col
        ):
            continue
        new_col = col
        for pattern, replacement in combined_replace_dict.items():
            new_col = re.sub(pattern, replacement, new_col)
        new_col_dict[col] = new_col

    # Replace column names according to constructed replace dict
    df_clean = df.selectExpr(
        *[f" `{old_col}` as `{new_col}`" for old_col, new_col in new_col_dict.items()]
    )
    return df_clean


def check_null_values(df: DataFrame) -> None:
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