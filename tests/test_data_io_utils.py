"""Tests data I/O functions."""
import unittest
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("src")

from src.utils.data_io import write_json, read_json


class TestJsonFunctions(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and files for testing."""
        self.temp_dir = TemporaryDirectory()
        self.sample_data = {"key": "value"}  # Sample data for writing and reading

        # Create the temp directory under model_artifacts
        self.fpath = Path(self.temp_dir.name) / "test.json"

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_write_json(self):
        """Test write_json function."""
        # Write data to features_targets.json
        write_json(self.fpath, self.sample_data, overwrite=False)

        self.assertTrue(self.fpath.exists(), f"{self.fpath} should exist.")

        # Verify that the file content is correct
        with self.fpath.open("r") as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, self.sample_data, "Written data should match the expected data.")

    def test_write_json_no_overwrite(self):
        """Test write_json when file exists and overwrite=False."""
        write_json(self.fpath, self.sample_data, overwrite=False)

        # Try writing again without overwrite and check if it raises FileExistsError
        with self.assertRaises(FileExistsError):
            write_json(self.fpath, self.sample_data, overwrite=False)

    def test_write_json_with_overwrite(self):
        """Test write_json when file exists and overwrite=True."""
        write_json(self.fpath, self.sample_data, overwrite=False)

        # Write new data to overwrite the file
        new_data = {"new_key": "new_value"}
        write_json(self.fpath, new_data, overwrite=True)

        with self.fpath.open("r") as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, new_data, "File should be overwritten with new data.")

    def test_read_json(self):
        """Test read_json function."""
        # Write the file first
        write_json(self.fpath, self.sample_data, overwrite=False)

        # Now read the file
        loaded_data = read_json(self.fpath)

        # Verify the content is correct
        self.assertEqual(loaded_data, self.sample_data, "Read data should match the written data.")

    def test_read_json_file_not_found(self):
        """Test read_json when the file does not exist."""
        # Attempt to read a file that does not exist
        with self.assertRaises(FileNotFoundError):
            read_json(Path(self.temp_dir.name) / "non_existent_file.json")

    def test_write_json_creates_directories(self):
        """Test that directories are created when writing a JSON file."""
        self.assertFalse(self.fpath.exists(), "File should not exist before writing.")

        # Write the mappings.json file
        write_json(self.fpath, self.sample_data, overwrite=False)

        # Check if the directories were created
        self.assertTrue(self.fpath.exists(), "File should exist after writing.")


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
