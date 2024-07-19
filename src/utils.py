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
    from_dow = pd.Timestamp(day=1, month=1, year=year_from).day_of_week
    to_dow = pd.Timestamp(day=1, month=1, year=year_to).day_of_week

    offset = to_dow - from_dow

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
):
    dow_offset = day_of_week_offset(year_from=from_year, year_to=to_year)

    pass
