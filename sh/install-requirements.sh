#!/bin/sh -x

# This script does all the initial setup needed to be able
# to authenticate requests to the RA private repo, then
# installs all requirements from there and PyPI as specified
# in requirements.txt

if [ "$#" -eq  "1" ]
  then
     suffix="-$1"
 else
     suffix=""
 fi

# Upgrade pip from the default cluster version to a newer
# version that supports keyring.
pip install --upgrade pip

# Install keyring support and support for a keyring that
# knows how to auth to google.
pip install keyring keyrings-google-artifactregistry-auth

# Install all our requirements from requirements.txt, looking first
# in the RA private repo then in public PyPI.
pip install \
  --index-url https://us-central1-python.pkg.dev/cube-machine-learning/rewiring-america-python-repo/simple \
  --extra-index-url https://pypi.org/simple/ \
  --keyring-provider import \
  -r "../requirements${suffix}.txt"
