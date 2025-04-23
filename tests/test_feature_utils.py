"""Tests feature transformation utility functions."""

from functools import reduce
import os
import sys
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from pyspark.sql import DataFrame

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("../src")
    from feature_utils import apply_upgrades, create_string_indexer, fill_null_with_column


class TestCreateStringIndexer(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_string_indexer(self):
        """Test string indexer function."""
        # Create a sample DataFrame
        data = [("apple",), ("banana",), ("cherry",), ("apple",)]
        columns = ["fruit"]
        df = spark.createDataFrame(data, columns)

        # Call the function
        indexer = create_string_indexer(df, "fruit")

        # Transform the DataFrame
        transformed_df = indexer.transform(df)

        # Collect the results
        result = transformed_df.select("fruit", "fruit_index").distinct().collect()

        # Check that each unique fruit has a unique index
        unique_indices = set(row["fruit_index"] for row in result)
        self.assertEqual(len(unique_indices), 3)  # Should match the number of unique fruit names

        # Ensure that indices are assigned in alphabetical order (due to stringOrderType="alphabetAsc")
        expected_mapping = {"apple": 0.0, "banana": 1.0, "cherry": 2.0}
        for row in result:
            self.assertEqual(row["fruit_index"], expected_mapping[row["fruit"]])


class ApplyUpgrades(unittest.TestCase):
    """Test feature upgrade tranformations."""

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_apply_upgrades(self):
        """Test feautre upgrade tranformations match expected output."""
        baseline_features = pd.read_csv(
            "test_baseline_features_input.csv", keep_default_na=False, na_values=[""]
        ).reset_index()
        df_out_expected = pd.read_csv("test_upgraded_features.csv", keep_default_na=False, na_values=[""])

        # apply upgrades to baseline features by upgrade group
        # and put rows back in the same order as they were read in
        df_out = (
            reduce(
                DataFrame.unionByName,
                [
                    apply_upgrades(
                        baseline_building_features=spark.createDataFrame(g),
                        upgrade_id=upgrade_id,
                    )
                    for upgrade_id, g in baseline_features.groupby("upgrade_id_input")
                ],
            )
            .toPandas()
            .sort_values("index")
            .drop("index", axis=1)
            .reset_index(drop=True)
        )

        # bool columns (prefixed 'has_' don't play well with nulls
        # so just maps nulls to False and so that we can compare
        has_cols = [col for col in df_out.columns if "has" in col]
        df_out[has_cols] = df_out[has_cols].fillna(False).astype(bool)
        df_out_expected[has_cols] = df_out_expected[has_cols].fillna(False).astype(bool)

        # check whether logic produced expected output on the same set of columns in same order
        # TODO: remove tolerance param after rounding is implemented in dmlutils upstream
        assert_frame_equal(df_out[df_out_expected.columns], df_out_expected, atol=1e-03)


class TestFillNullWithColumn(unittest.TestCase):

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_fill_null_with_column_basic(self):
        # Create test data
        data = [(1, "X", "A", None), (2, "Y", None, "B"), (3, "Z", None, None), (4, None, "D", "C")]
        columns = ["id", "source_col", "col1", "col2"]
        df = spark.createDataFrame(data, columns)

        # Apply function
        result = fill_null_with_column(df, "source_col", ["col1", "col2"])

        # Expected data
        expected_data = [(1, "X", "A", "X"), (2, "Y", "Y", "B"), (3, "Z", "Z", "Z"), (4, None, "D", "C")]
        expected_df = spark.createDataFrame(expected_data, columns)

        # Compare results
        self.assertEqual(result.collect(), expected_df.collect())

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_fill_null_with_column_empty_list(self):
        # Data with empty columns_to_fill list
        data = [(1, "A", None), (2, None, "B")]
        columns = ["source_col", "id", "col1"]
        df = spark.createDataFrame(data, columns)

        # Apply function with empty list
        result = fill_null_with_column(df, "source_col", [])

        # Should be unchanged
        expected_df = df
        self.assertEqual(result.collect(), expected_df.collect())


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
