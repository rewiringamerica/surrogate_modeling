"""Test constants."""

import os
import sys

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("../src")

import unittest

from src import constants


class ConstantTestCase(unittest.TestCase):
    """Test various constants."""

    def test_kw_btu(self):
        """Test conversions between kwh and btu."""
        self.assertGreater(
            constants.KILOWATT_HOUR_TO_BRITISH_THERMAL_UNIT,
            1.0,
            "A kwh is more than a BTU.",
        )

        self.assertLess(
            constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR,
            1.0,
            "A BTU is less than a kwh.",
        )

        self.assertAlmostEqual(
            1.0,
            constants.KILOWATT_HOUR_TO_BRITISH_THERMAL_UNIT
            * constants.BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR,
            places=10,
            msg="Conversions between btu and kwh should be recipricals.",
        )

    def test_lb_to_kg(self):
        """Test conversions between kwh and btu."""
        self.assertGreater(
            constants.KILOGRAM_TO_POUND, 1.0, "A pound is more than a kilo."
        )

        self.assertLess(
            constants.POUND_TO_KILOGRAM, 1.0, "A kilo is less than a pound."
        )

        self.assertAlmostEqual(
            1.0,
            constants.KILOGRAM_TO_POUND * constants.POUND_TO_KILOGRAM,
            places=10,
            msg="Conversions between pounds and kilos should be recipricals.",
        )


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
