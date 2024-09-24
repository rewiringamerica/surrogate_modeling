"""Tests to make sure we depend on dmutils properly."""

import os
import sys
import unittest
import subprocess

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    sys.path.append("../src")

    # If your cluster is 15.0.0 or higher, you should install
    # requirements-test.txt at the cluster level by uploading
    # it. This approach is for older clusters.
    db_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    major = int(db_version.split(".")[0])
    if major < 15:
        subprocess.call(["/bin/sh", "../sh/install-requirements.sh", "test"])


import dmlutils


class DmUtilsTestCase(unittest.TestCase):
    """Test access to dependency `dmutils`."""

    def test_version(self):
        """Test accessing `dmutils.version`."""
        version = dmlutils.version

        self.assertIsNotNone(version)
        self.assertEqual(str, type(version))


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
