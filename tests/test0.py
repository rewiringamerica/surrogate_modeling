"""Test cases to try out integration of VSCode and Databricks."""

import unittest
import sys
import os

class MyTestCase(unittest.TestCase):
    """My test case."""
    def test_true(self):
        """Test something that is true."""
        self.assertTrue(True)

class MyOtherTestCase(unittest.TestCase):
    """My other test case."""
    def test_equals(self):
        """Test some other thing that is true."""
        self.assertEqual(4, 2 + 2)


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
