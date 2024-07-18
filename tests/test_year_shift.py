"""Test cases for year shifting code."""

import unittest
import sys
import os

class MyTestCase(unittest.TestCase):

    def test_true(self):
        self.assertTrue(True)

class MyOtherTestCase(unittest.TestCase):

    def test_true(self):
        self.assertTrue(True)


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instatiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()