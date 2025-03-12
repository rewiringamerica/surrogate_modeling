import json
from pathlib import Path


def write_json(fpath, data, overwrite=False):
    """Write a JSON file for a given version, with an overwrite check.

    Args:
        fpath (str): The name of the JSON file (e.g., "features_targets.json").
        data (dict): The dictionary to save as JSON.
        overwrite (bool): If False, raises an error if the file already exists.

    Raises:
        FileExistsError: If the file exists and overwrite=False.
    """

    if fpath.exists() and not overwrite:
        raise FileExistsError(f"{fpath} already exists. Use overwrite=True to replace it.")

    # Write JSON file
    with fpath.open("w") as f:
        json.dump(data, f, indent=4)

    print(f"{fpath} saved")


def read_json(fpath):
    """Read a JSON file for a given version.

    Args:
        fpath (str): The path of the JSON file.

    Returns:
        dict: The loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist.")

    with fpath.open("r") as f:
        return json.load(f)
