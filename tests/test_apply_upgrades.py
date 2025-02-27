"""Tests feature upgrade function."""

from functools import reduce
import os
import sys
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from pyspark.sql import DataFrame

from dmlutils.surrogate_model.apply_upgrades import (
    read_test_baseline_inputs, 
    read_test_upgraded_outputs
)

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("src")
    from src.feature_utils import apply_upgrades

class ApplyUpgrades(unittest.TestCase):
    """Test feautre upgrade tranformations."""

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_apply_upgrades(self):
        """Test feautre upgrade tranformations match expected output."""
        baseline_features = read_test_baseline_inputs()
        df_out_expected = read_test_upgraded_outputs()

        # apply upgrades to baseline features by upgrade group
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
            .reset_index(drop=True)
        )

        # bool columns (prefixed 'has_' don't play well with nulls
        # so just maps nulls to False and so that we can compare
        has_cols = [col for col in df_out.columns if "has" in col]
        df_out[has_cols] = df_out[has_cols].fillna(False).astype(bool)
        df_out_expected[has_cols] = df_out_expected[has_cols].fillna(False).astype(bool)

        # check whether logic produced expected output on the same set of columns
        print(df_out_expected.columns)
        assert_frame_equal(df_out, df_out_expected[df_out.columns])


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
