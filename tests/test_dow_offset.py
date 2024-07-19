"""Tests for day of week shifting logic."""

import unittest
from typing import Set
import os
import sys
from pathlib import Path

import pandas as pd

# TODO: Where do we configure this so we don't have to put this here?
sys.path.append(str((Path().parent / "src").absolute()))

import utils


class DayOfWeekOffsetTest(unittest.TestCase):
    """Test our ability to find the right day offset from year to year."""

    def test_same_year(self):
        """Test there is zero offset from a year to itself."""
        for year in range(1990, 2031):
            offset = utils.day_of_week_offset(year_from=year, year_to=year)
            self.assertEqual(0, offset)

    def test_known_years(self):
        """Test for a couple of years where we know the offset."""
        # 2006 started on a Sunday
        # 2007 stated on a Monday
        offset_2007_to_2006 = utils.day_of_week_offset(year_from=2007, year_to=2006)
        self.assertEqual(-1, offset_2007_to_2006)

        offset_2006_to_2007 = utils.day_of_week_offset(year_from=2006, year_to=2007)
        self.assertEqual(1, offset_2006_to_2007)

    def test_negative_direction(self):
        """Test that offsets in opposite directions are negatives of one another."""
        for year_from in range(1990, 2031):
            for year_to in range(1990, 2031):
                offset_from_to = utils.day_of_week_offset(
                    year_from=year_from, year_to=year_to
                )
                offset_to_from = utils.day_of_week_offset(
                    year_from=year_to, year_to=year_from
                )

                self.assertEqual(offset_from_to, -offset_to_from)

    def test_day_in_year(self):
        """Test that when we offset a day in a year it ends up in the right place."""
        target_year = 2006
        target_month = 2
        target_day = 14

        target_timestamp = pd.Timestamp(
            month=target_month, day=target_day, year=target_year
        )

        offsets_seen: Set[int] = set()

        for year in range(1990, 2031):

            # Same date in the given year.
            timestamp = pd.Timestamp(month=target_month, day=target_day, year=year)

            self.assertEqual(target_timestamp.month, timestamp.month)
            self.assertEqual(target_timestamp.day, timestamp.day)

            offset = utils.day_of_week_offset(year_from=year, year_to=target_year)

            offsets_seen.add(offset)

            offset_timestamp = timestamp + pd.Timedelta(days=offset)

            self.assertEqual(target_timestamp.day_of_week, offset_timestamp.day_of_week)

        # Make sure we saw all offsets.
        self.assertSetEqual(set(range(-3, 4)), offsets_seen)


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
