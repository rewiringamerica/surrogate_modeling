# TODO: delete this file once dmutils is updated
import pandas as pd
import re
from src.dmutils import constants

EER_CONVERSION = {
    "EER": 1.0,
    "SEER": constants.SEER_TO_EER,
    "EER2": constants.EER2_TO_EER,
    "SEER2": constants.EER2_TO_EER * constants.SEER_TO_EER,
}

# list of columns containing fuel types for appliances
APPLIANCE_FUEL_COLS = [
    "clothes_dryer_fuel",
    "cooking_range_fuel",
    "heating_fuel",
    "hot_tub_spa_fuel",
    "pool_heater_fuel",
    "water_heater_fuel",
]

# list of columns containing fuel types for appliances
GAS_APPLIANCE_INDICATOR_COLS = [
    "has_gas_fireplace",
    "has_gas_grill",
    "has_gas_lighting",
]

# Mapping from climate zone to insulation values for basic enclosure
BASIC_ENCLOSURE_INSULATION = pd.DataFrame(
    {
        "climate_zone_temp": [1, 2, 3, 4, 5, 6, 7],
        "existing_insulation_max_threshold": [13, 30, 30, 38, 38, 38, 38],
        "insulation_upgrade": [30, 49, 49, 60, 60, 60, 60],
    }
)


def extract_r_value(construction_type: str, set_none_to_inf: bool = False) -> int:
    """
    Extract R-value from an unformatted string.

    Assumption: all baseline walls have similar R-value of ~4.
    The returned value is for additional insulation only. Examples:
        Uninsulated brick, 3w, 12": ~4
            (https://ncma.org/resource/rvalues-of-multi-wythe-concrete-masonry-walls/)
        Uninsulated wood studs: ~4
            (assuming 2x4 studs and 1.25/inch (air gap has higher R-value than wood), 3.5*1.25=4.375)
        Hollow Concrete Masonry Unit, Uninsulated: ~4 per 6"
            (https://ncma.org/resource/rvalues-ufactors-of-single-wythe-concrete-masonry-walls/)

    >>> extract_r_value('Finished, R-13')
    13
    >>> extract_r_value('Brick, 12-in, 3-wythe, R-15')
    15
    >>> extract_r_value('2ft R10 Under, Horizontal')
    10
    >>> extract_r_value('R-5, Exterior')
    5
    >>> extract_r_value('Ceiling R-19')
    19
    >>> extract_r_value('CMU, 6-in Hollow, Uninsulated')
    0
    >>> extract_r_value('None')
    0
    >>> extract_r_value('None', set_none_to_inf=True)
    999

    Parameters
    ----------
    construction_type
        The wall insulation type of a building.

    set_none_to_inf
        A flag that indicates whether to map inputs of 'None' to np.inf.

    Returns
    -------
    int
        The extracted R-value.

    Raises
    ------
    ValueError
        If the R-value of the construction type cannot be determined.
    """
    if "uninsulated" in construction_type.lower():
        return 0
    if construction_type.lower() == "none":
        if set_none_to_inf:
            return 999
        else:
            return 0
    m = re.search(r"\br-?(\d+)\b", construction_type, flags=re.I)
    if not m:
        raise ValueError(f"Cannot determine R-value of the construction type: {construction_type}")
    return int(m.group(1))


def extract_cooling_efficiency(cooling_efficiency: str) -> float:
    """
    Convert a ResStock cooling efficiency into EER value.

    >>> extract_cooling_efficiency('AC, SEER 13') / EER_CONVERSION['SEER']
    13.0
    >>> extract_cooling_efficiency('ASHP, SEER 20, o7.7 HSPF') / EER_CONVERSION['SEER']
    20.0
    >>> extract_cooling_efficiency('Room AC, EER 10.7')
    10.7
    >>> extract_cooling_efficiency('Shared Cooling')
    13.0
    >>> extract_cooling_efficiency('None') >= 99
    True

    Parameters
    ----------
    cooling_efficiency
        The cooling efficiency of a ResStock sample building.

    Returns
    -------
    float
        The EER value of the cooling efficiency.

    Raises
    ------
    ValueError
        If the cooling efficiency cannot be extracted.
    """
    if cooling_efficiency == "None":
        # "infinitely" high efficiency to mimic a nonexistent cooling
        return 999.0
    # mapping of HVAC Shared Efficiencies -> cooling_system_cooling_efficiency from options.tsv
    if cooling_efficiency == "Shared Cooling":
        cooling_efficiency = "SEER 13"

    m = re.search(r"\b(SEER2|SEER|EER)\s+(\d+\.?\d*)", cooling_efficiency)
    if m:
        try:
            return EER_CONVERSION[m.group(1)] * float(m.group(2))
        except (ValueError, KeyError):
            raise ValueError(f"Cannot extract cooling efficiency from: {cooling_efficiency}")


def extract_heating_efficiency(heating_efficiency: str) -> int:
    """
    Extract heating efficiency from string and convert to nominal percent efficiency.

    Source: https://www.energyguru.com/EnergyEfficiencyInformation.htm

    >>> extract_heating_efficiency('Fuel Furnace, 80% AFUE')
    .8
    >>> extract_heating_efficiency('ASHP, SEER 15, 8.5 HSPF')
    2.49
    >>> extract_heating_efficiency('None') >= 9
    True
    >>> extract_heating_efficiency('Electric Baseboard, 100% Efficiency')
    1.0
    >>> extract_heating_efficiency('Fan Coil Heating And Cooling, Natural Gas')
    .78
    >>> extract_heating_efficiency('Other')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Cannot extract heating efficiency from: ...

    Parameters
    ----------
    heating_efficiency
        The heating efficiency of a ResStock sample building.

    Returns
    -------
    float
        The nominal percent value of the heating efficiency.

    Raises
    ------
    ValueError
        If the heating efficiency cannot be extracted.
    """
    efficiency = heating_efficiency.rsplit(", ", 1)[-1]
    if efficiency == "None":
        return 9.0
    # mapping of HVAC Shared Efficiencies -> heating_system_heating_efficiency from options.tsv
    if efficiency == "Electricity":
        return 1.0
    if efficiency in ["Fuel Oil", "Natural Gas", "Propane"]:
        return 0.78

    try:
        number = float(efficiency.strip().split(" ", 1)[0].strip("%"))
    except ValueError:
        return None

    if efficiency.endswith("AFUE"):
        return number / 100
    if efficiency.endswith("HSPF"):
        return round(number / (constants.KILOWATT_HOUR_TO_BRITISH_THERMAL_UNIT / 1000), 3)

    # 'Other' - e.g. wood stove - is not supported
    return number / 100
