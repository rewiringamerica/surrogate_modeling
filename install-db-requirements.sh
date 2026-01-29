#!/bin/bash
#
# This script is meant to be used as an init script on a Databricks 14.3 LTS
# cluster.

echo "Running install-db-requirements.sh init script."


if [[ $DATABRICKS_RUNTIME_VERSION != "14.3" ]]; then
    # Write error message to both stdout and stderr
    echo "Expected Databricks runtime 14.3, but found ${DATABRICKS_RUNTIME_VERSION}"
    echo "Expected Databricks runtime 14.3, but found ${DATABRICKS_RUNTIME_VERSION}" >&2
    exit 1
fi

echo "Databricks runtime version looks correct: ${DATABRICKS_RUNTIME_VERSION}"


TEMP_REQUIREMENTS=$(mktemp /tmp/requirements.txt.XXXXXX)
cat >"$TEMP_REQUIREMENTS" <<'EOF'
# The contents of this section should be a copy of requirements-db-14.3.txt.
# That way this is a self-contained shell script that can install requirements
# on a Databricks cluster as an init script.
--extra-index-url https://us-central1-python.pkg.dev/cube-machine-learning/rewiring-america-python-repo/simple

appnope==0.1.4 ; (sys_platform == "darwin" or platform_system == "Darwin") and python_version == "3.10"
cfgv==3.5.0 ; python_version == "3.10"
colorama==0.4.6 ; (platform_system == "Windows" or sys_platform == "win32") and python_version == "3.10"
cython==0.29.32 ; python_version == "3.10"
db-dtypes==1.4.2 ; python_version == "3.10"
distro==1.9.0 ; python_version == "3.10"
dmlbootstrap==0.3.0 ; python_version == "3.10"
dmlutils==0.13.6 ; python_version == "3.10"
duckdb==1.2.2 ; python_version == "3.10"
eemeter==3.2.0 ; python_version == "3.10"
eeweather==0.3.28 ; python_version == "3.10"
et-xmlfile==2.0.0 ; python_version == "3.10"
flask==2.2.5 ; python_version == "3.10"
gcsfs==2023.6.0 ; python_version == "3.10"
geocoder==1.38.1 ; python_version == "3.10"
gitpython==3.1.27 ; python_version == "3.10"
google-cloud-bigquery-storage==2.31.0 ; python_version == "3.10"
google-cloud-bigquery==3.30.0 ; python_version == "3.10"
google-cloud-secret-manager==2.23.3 ; python_version == "3.10"
greenlet==3.2.1 ; python_version == "3.10" and (platform_machine == "aarch64" or platform_machine == "ppc64le" or platform_machine == "x86_64" or platform_machine == "amd64" or platform_machine == "AMD64" or platform_machine == "win32" or platform_machine == "WIN32")
grpc-google-iam-v1==0.14.2 ; python_version == "3.10"
h3==4.2.2 ; python_version == "3.10"
identify==2.6.16 ; python_version == "3.10"
imagehash==4.3.1 ; python_version == "3.10"
jinja2==3.1.2 ; python_version == "3.10"
jupyter-core==5.2.0 ; python_version == "3.10"
keyrings-google-artifactregistry-auth==1.1.2 ; python_version == "3.10"
lazr-restfulclient==0.14.4 ; python_version == "3.10"
lazr-uri==1.0.6 ; python_version == "3.10"
lazy-loader==0.3 ; python_version == "3.10"
mako==1.2.0 ; python_version == "3.10"
markdown==3.4.1 ; python_version == "3.10"
markupsafe==2.1.1 ; python_version == "3.10"
notebook-shim==0.2.2 ; python_version == "3.10"
openpyxl==3.1.5 ; python_version == "3.10"
pillow==9.4.0 ; python_version == "3.10"
pre-commit==4.5.1 ; python_version == "3.10"
probableparsing==0.0.1 ; python_version == "3.10"
proto-plus==1.26.1 ; python_version == "3.10"
py4j==0.10.9.7 ; python_version == "3.10"
py==1.11.0 ; python_version == "3.10" and implementation_name == "pypy"
pygments==2.11.2 ; python_version == "3.10"
pyjwt==2.3.0 ; python_version == "3.10"
pynacl==1.5.0 ; python_version == "3.10"
pyproj==3.7.1 ; python_version == "3.10"
pyspark==3.5.5 ; python_version == "3.10"
python-crfsuite==0.9.11 ; python_version == "3.10"
pywavelets==1.4.1 ; python_version == "3.10"
pywin32-ctypes==0.2.3 ; python_version == "3.10" and sys_platform == "win32"
pywin32==310 ; sys_platform == "win32" and platform_python_implementation != "PyPy" and python_version == "3.10"
pywinpty==2.0.15 ; python_version == "3.10" and os_name == "nt"
pyyaml==6.0 ; python_version == "3.10"
radbcluster==14.3.1 ; python_version == "3.10"
ratelim==0.1.6 ; python_version == "3.10"
secretstorage==3.3.1 ; python_version == "3.10"
send2trash==1.8.0 ; python_version == "3.10"
shapely==2.1.0 ; python_version == "3.10"
sqlalchemy==1.4.39 ; python_version == "3.10"
timezonefinder==6.5.7 ; python_version == "3.10"
toml==0.10.2 ; python_version == "3.10"
typing-extensions==4.4.0 ; python_version == "3.10"
usaddress-scourgify==0.6.0 ; python_version == "3.10"
usaddress==0.5.13 ; python_version == "3.10"
usingversion==0.1.2 ; python_version == "3.10"
werkzeug==2.2.2 ; python_version == "3.10"
yaml-config==0.1.5 ; python_version == "3.10"

EOF

echo "TEMP_REQUIREMENTS"
echo "$TEMP_REQUIREMENTS contains:"
echo ------------------
cat "$TEMP_REQUIREMENTS"
echo ------------------

echo
echo

echo "PIP LIST BEFORE INSTALL"
python -m pip list

echo
echo

echo "UPGRADE PIP AND ADD KEYRING SUPPORT"

# Upgrade pip from the default cluster version to a newer
# version that supports keyring.
python -m pip install --upgrade pip

# Install keyring support and support for a keyring that
# knows how to auth to google.
python -m pip install keyring keyrings-google-artifactregistry-auth

echo
echo

echo "PIP INSTALL"
python -m pip install \
  --index-url https://us-central1-python.pkg.dev/cube-machine-learning/rewiring-america-python-repo/simple \
  --extra-index-url https://pypi.org/simple/ \
  --keyring-provider import \
  -r "$TEMP_REQUIREMENTS"

echo
echo

echo "PIP LIST AFTER INSTALL"
python -m pip list
