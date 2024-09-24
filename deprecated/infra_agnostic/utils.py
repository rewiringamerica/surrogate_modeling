import glob
import logging
import os
import tempfile

import pandas as pd


def file_cache(cache_path=None):
    """A file caching decorator

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
        f = None  # function to be cached

        def __init__(self, function):
            self.f = function
            self.func_cache_path = os.path.join(cache_path, function.__name__)
            if not os.path.isdir(self.func_cache_path):
                os.mkdir(self.func_cache_path)

        def __call__(self, *args):
            # function name is to have a meaningful key even with empty args
            arg_key = "__".join(str(arg) for arg in (self.f.__name__, *args))
            cache_fpath = os.path.join(self.func_cache_path, arg_key + ".pq")
            if os.path.exists(cache_fpath):
                return pd.read_parquet(cache_fpath)
            res = self.f(*args)
            res.to_parquet(cache_fpath)
            return res

        def reset_cache(self):
            for fpath in glob.glob(os.path.join(self.func_cache_path, "*")):
                os.remove(fpath)

    return Decorator
