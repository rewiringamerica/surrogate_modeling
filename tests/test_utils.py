"""Tests utility functions."""

import os
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
import subprocess
import unittest

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("../src")

import util
import constants


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
        test_output = util.clean_columns(
            df=test_input,
            remove_columns_with_substrings=["remove__me"],
            remove_substrings_from_columns=["shorter_pls"],
            replace_column_substrings_dict={"natural_gas": "methane_gas"},
        )
        self.assertCountEqual(
            test_output.columns, ["building__id", "make_me_", "fuck_methane_gas"]
        )
        # check that if we pass no args, we get back the identical schema
        test_output_no_change = util.clean_columns(
            df=test_input, replace_period_character="."
        )
        self.assertCountEqual(test_output_no_change.columns, test_input.columns)

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_clean_bsb_output_cols(self):
        """Test column cleaning."""
        test_input = spark.createDataFrame(
            pd.DataFrame(
                {
                    "building_id": [0.0],
                    "job_id": [0.0],
                    "started_at": ["0"],
                    "completed_at": ["0"],
                    "completed_status": ["0"],
                    "step_failures": [0.0],
                    "report_simulation_output_emissions_x": [0.0],
                    "report_simulation_output_system_use_x": [0.0],
                    "report_simulation_output_include_x": [0.0],
                    "apply_upgrade_x": [0.0],
                    "utility_bills_x": [0.0],
                    "applicable_x": [0.0],
                    "upgrade_costs_x": [0.0],
                    "add_timeseries_x": [0.0],
                    "output_format_x": [0.0],
                    "qoi_report_qoi_specific_qoi": [0.0],
                    "end_use_specific_end_use": [0.0],
                    "fuel_use_specific_fuel_use": [0.0],
                    "x_report_simulation_output_x": [0.0],
                    "x_m_btu_x": [0.0],
                    "x_heat_pump_backup_x": [0.0],
                    "x_energy_use_x": [0.0],
                }
            )
        )

        # check that the columns of the output dataframe are as expected
        test_output = util.clean_bsb_output_cols(bsb_df=test_input)
        self.assertCountEqual(
            test_output.columns,
            [
                "building_id",
                "specific_qoi",
                "specific_end_use",
                "specific_fuel_use",
                "x_out_x",
                "x_energy_consumption_m_btu_x",
                "x_hp_bkup_x",
                "x_site_energy_x",
            ],
        )

    @unittest.skipIf(
        os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is None,
        reason="Only runs on databricks cluster.",
    )
    def test_convert_column_units(self):
        """Test function for unit conversion across dataframe."""
        test_input = spark.createDataFrame(
            pd.DataFrame(
                {
                    "energy_m_btu": [1e-6],
                    "kgco2_lb": [1.0],
                    "power_btu_h": [1.0],
                    "temp_f": [33.0],
                }
            )
        )

        # apply conversion
        test_output_df = util.convert_column_units(test_input)
        # transform df into a dictionary for easier access
        test_output_dict = test_output_df.collect()[0].asDict()
        # check that the column names and values were transformed as expected
        self.assertAlmostEqual(
            test_output_dict["energy_kwh"],
            constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR,
        )
        self.assertAlmostEqual(
            test_output_dict["kgco2_kg"], constants.POUND_TO_KILOGRAM
        )
        self.assertAlmostEqual(
            test_output_dict["power_kw"],
            constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR,
        )
        self.assertAlmostEqual(test_output_dict["temp_c"], 5 / 9)


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    )
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
