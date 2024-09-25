import itertools
import math
import re
from typing import Dict
import re
import pandas as pd

# Define BTU_PER_WH if not already defined
BTU_PER_WH = 3.412  # Example value, adjust as necessary

SEER_TO_EER = 0.875


def convert_heating_efficiency(value):
    """convert HSPF to percentage or extract percentage directly"""
    if "HSPF" in value:
        # Extract the numeric value and convert HSPF to percentage
        hspf_value = float(re.search(r"(\d+\.?\d+) HSPF", value).group(1))
        return hspf_value * 100 / BTU_PER_WH
    else:
        # Extract percentage directly if present
        match = re.search(r"(\d+\.?\d+)%", value)
        return float(match.group(1)) if match else None


# Extract SEER, HSPF, and AFUE ratings from 'in_hvac_heating_efficiency'
def extract_seer(value):
    """Extract SEER ratings from 'in_hvac_heating_efficiency' values"""
    parts = value.split(", ")
    if len(parts) > 1 and "SEER" in parts[1]:
        seer_str = value.split(", ")[1]  # Extract the SEER substring
        seer_rating = seer_str.split(" ")[1]  # Extract the SEER rating value
        return float(seer_rating)
    return None


def extract_hspf(value):
    """Extract HSPF ratings from 'in_hvac_heating_efficiency' values"""
    parts = value.split(", ")
    if len(parts) > 2 and "HSPF" in parts[2]:
        return float(parts[2].split(" HSPF")[0])
    return None


def extract_afue(value):
    """Extract AFUE ratings from 'in_hvac_heating_efficiency' values"""
    parts = value.split(", ")
    if len(parts) > 1 and "%" in parts[1]:
        return float(parts[1].split("%")[0])
    return None


def extract_cooling_efficiency(text):
    """Converts SEER ratings to EER values"""
    if pd.isna(text):
        return 99
    match = re.match(r"((?:SEER|EER))\s+([\d\.]+)", text)
    if match:
        efficiency_type, value = match.groups()
        if efficiency_type == "SEER":
            value = float(value) * SEER_TO_EER
        else:
            value = float(value)
        return value
    else:
        return 99


def vintage2age2010(vintage: str) -> int:
    """vintage of the building in the year of 2010
    >>> vintage2age2000('<1940')
    80
    >>> vintage2age2000('1960s')
    50
    """
    vintage = vintage.strip()
    if vintage.startswith("<"):  # '<1940' bin in resstock
        return 80
    else:
        return 2010 - int(vintage[:4])
