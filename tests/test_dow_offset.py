"""Tests for day of week shifting logic."""

import unittest
from typing import Set
import os
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_series_equal


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
        self.assertEqual(1, offset_2007_to_2006)

        offset_2006_to_2007 = utils.day_of_week_offset(year_from=2006, year_to=2007)
        self.assertEqual(-1, offset_2006_to_2007)

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

            offset_timestamp = (
                timestamp
                + pd.DateOffset(days=offset)
                + pd.DateOffset(years=target_year - year)
            )

            self.assertEqual(timestamp.day_of_week, offset_timestamp.day_of_week)

        # Make sure we saw all offsets.
        self.assertSetEqual(set(range(-3, 4)), offsets_seen)


class ShiftYearTestCase(unittest.TestCase):
    """Test shifting a whole year worth of dates."""

    def setUp(self) -> None:
        # 2006 started on a Sunday
        self.ref_year = 2006

        # 2019 started on a Tuesday
        self.from_year = 2019
        jan_1_from_year = pd.Timestamp(year=self.from_year, month=1, day=1)

        self.df_dates = pd.DataFrame(
            [
                {
                    f"date_{self.from_year}": jan_1_from_year + pd.DateOffset(days=day),
                    f"day_of_{self.from_year}": day,
                }
                for day in range(0, 365)
            ]
        )

    def test_shift_to(self):
        """Test shifting a year full of dates to a reference year."""
        dates_ref_year = utils.shift_year_preserve_dow(
            self.df_dates[f"date_{self.from_year}"],
            from_year=self.from_year,
            to_year=self.ref_year,
        )

        # All shifted dates are unique.
        self.assertEqual(365, len(dates_ref_year.unique()))

        # Day of week should not change after the shift. Assert across the series.
        assert_series_equal(
            dates_ref_year.dt.day_of_week,
            self.df_dates[f"date_{self.from_year}"].dt.day_of_week,
        )

        # Since we shifted forward two days when we moved year (Sunday to Tuesday),
        # the min shifted date should be Jan 3, and the max shifted date should be
        # Jan 2 of the following year.

        self.assertEqual(
            pd.Timestamp(month=1, day=3, year=self.ref_year), dates_ref_year.min()
        )
        self.assertEqual(
            pd.Timestamp(month=1, day=2, year=self.ref_year + 1), dates_ref_year.max()
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
