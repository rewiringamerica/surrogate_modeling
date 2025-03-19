"""Tests utility functions."""

import os
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
import subprocess
import unittest

from dmlutils import constants

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("../src")

from utils import data_cleaning

# TODO: Remove skips for unit test once we have spark testing working on git:
# https://www.notion.so/rewiringamerica/Local-Spark-Testing-4aef885e20034c18b1a2fba6c355e82c?pvs=4


class ResStockDataTestCase(unittest.TestCase):
    """Test functionality of ResStock processing functions."""

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_clean_columns(self):
        """Test column cleaning."""
        test_input = spark.createDataFrame(
            pd.DataFrame(
                {
                    "building.id": [0.0],
                    "remove.me": [0.0],
                    "make_me_shorter_pls": [1.0],
                    "fuck_natural_gas": [0.0],
                }
            )
        )

        # check the each of the the supported operations
        test_output = data_cleaning.edit_columns(
            df=test_input,
            remove_columns_with_substrings=["remove__me"],
            remove_substrings_from_columns=["shorter_pls"],
            replace_column_substrings_dict={"natural_gas": "methane_gas"},
        )
        self.assertCountEqual(test_output.columns, ["building__id", "make_me_", "fuck_methane_gas"])
        # check that if we pass no args, we get back the identical schema
        test_output_no_change = data_cleaning.edit_columns(df=test_input, replace_period_character=".")
        self.assertCountEqual(test_output_no_change.columns, test_input.columns)


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
