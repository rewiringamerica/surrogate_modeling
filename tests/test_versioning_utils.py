"""Tests versioning functions."""
import os
import sys
import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import toml

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("src")

from src.versioning import get_poetry_version_no

class TestGetPoetryVersionNo(unittest.TestCase):
    
    @patch("builtins.open", new_callable=mock_open, read_data='''\
        [tool.poetry]
        version = "1.2.3"
        ''')
    @patch("toml.load")
    def test_get_poetry_version_no(self, mock_toml_load, mock_file):
        """Test zero-padding of poetry version number."""
        mock_toml_load.return_value = {"tool": {"poetry": {"version": "1.2.3"}}}

        with patch.object(Path, "parent", new_callable=lambda: Path("/fake/path")):
            result = get_poetry_version_no()

        self.assertEqual(result, "01_02_03")  # Expected zero-padded format

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()