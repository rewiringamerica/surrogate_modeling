import glob
import logging
import os
import tempfile

import pandas as pd


def file_cache(cache_path=None):
    """ A file caching decorator

    Assumptions about the underlying function:
        - all arguments are positional, and are strings
        - function returns a pandas DataFrame. Multiindexes are OK
        - if no cache_path provided, system $TMP will be used

    >>> @file_cache()
    ... def test_func():
    ...     print("Hello")
    ...     return pd.DataFrame({
    ...         'a': [1,2,3], 'b': [4,5,6], 'c': [5,7,9]
    ...     }).set_index(['a', 'b'])
    >>> test_func.reset_cache()
    >>> a = test_func()
    Hello
    >>> b = test_func()  # no output produced since the call is cached
    >>> (a == b).all().all()  # but the output is exactly the same
    True
    >>> test_func.reset_cache()
    """

    # if no cache_path provided, system $TMP will be used
    cache_path = cache_path or tempfile.gettempdir()
    if not os.path.isdir(cache_path):
        try:
            os.mkdir(cache_path)
        except OSError:
            logging.error(f"Failed to create cache directory {cache_path}")
            raise

    class Decorator:

        def __init__(self, function):
            self.f = function
            self.func_cache_path = os.path.join(cache_path, function.__name__)
            if not os.path.isdir(self.func_cache_path):
                os.mkdir(self.func_cache_path)

        def __call__(self, *args):
            # function name is to have a meaningful key even with empty args
            arg_key = '__'.join(str(arg) for arg in (self.f.__name__, *args))
            cache_fpath = os.path.join(
                self.func_cache_path, arg_key + '.pq')
            if os.path.exists(cache_fpath):
                return pd.read_parquet(cache_fpath)
            res = self.f(*args)
            res.to_parquet(cache_fpath)
            return res

        def reset_cache(self):
            for fpath in glob.glob(os.path.join(self.func_cache_path, '*')):
                os.remove(fpath)

    return Decorator


def day_of_week_offset(
        *,
        year_from: int,
        year_to: int        
) -> int:
    """
    Determing the offset in days between the same date in two
    different years.

    The idea is that if we take a date, say 02/14/2019 and shift it
    to a different year, like 2006, producing 02/14/2006, the day
    of the week may shift. In case, it moves from Thursday to Tuesday.
    
    If we want to maintain the same day of week, we would have to move
    2 days forward to 02/16/2006, which is a Thursday, like the original
    date of 02/14/2019 was. We call this number 2 the offset, and it is
    what this function computes. So the call

    .. code-block:: python

        offset(year_from=2019, year_to=2006)

    will return 2.

    Note that if an offset of 2 would work, so would an offset of 9 or 
    and offset of -5. Anything that adds or removes a multiple of 7 days
    will preserve the day of week.

    Since we would generally like the smallest maginitude offset possible,
    we always return an offset in the range from -3 to 3. Instead of returning
    -1, we could return 6, and it would produce the right day of the week,
    but since we prefer smaller magnitudes we will return -1.

    Parameters
    ----------
    year_from : int
        The year we are offsetting from, e.g. 2019
    year_to : int
        The year we are offsetting to, e.g. 2006.

    Returns
    -------
    int
        The offset, in days, between -3 and 3.
    """
    from_dow = pd.Timestamp(day=1, month=1, year=year_from).day_of_week
    to_dow = pd.Timestamp(day=1, month=1, year=year_to).day_of_week

    offset = from_dow - to_dow

    # Offset is now between -6 and -6. We want to 
    # normalize to between -3 and +3 so that we are
    # shifting days the minumum amount necessary to 
    # align days of the week.

    # Change the negative offsets to corresponding positive
    # ones by adding a week.
    offset = (offset + 7) % 7
    # Change the rollover point so we get -3 to +3
    offset = ((offset + 3) % 7) - 3

    return offset


def shift_year_preserve_dow(
        dates: pd.Series,
        *,
        from_year: int,
        to_year: int
) -> pd.Series:
    """
    Shift a series of dates from one year to another, preserving day of 
    week.

    This builds upon :py:func:`~day_of_week_offset`. 
    
    There are two steps:
    
    First, we shift the dates by the number of years between `from_year` and `to_year`. 
    Second, we compute the day of week offset between `from_year` and `to_year` and
    shift the dates by that many days to correct for the fact that when we may have
    altered the day of the week when we shifted the year.

    Note that we do not check that the original `dates` are in `from_year`.

    Parameters
    ----------
    dates : pd.Series
        The dates we want to shift.
    from_year : int
        The year we are shifting from.
    to_year : int
        The year we are shifting to.

    Returns
    -------
    pd.Series
        The shifted dates, preserving day of week.
    """
    dow_offset = day_of_week_offset(year_from=from_year, year_to=to_year)

    offset_by_days = dates + pd.DateOffset(days=dow_offset)

    offset_by_days_and_years = offset_by_days + pd.DateOffset(years=to_year - from_year)

    return offset_by_days_and_years
