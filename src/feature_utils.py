# TODO: Run tests using doctest

import re
from functools import reduce
from itertools import chain
from typing import Dict, Optional
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
    StringType,
    StructType,
    StructField,
)
from pyspark.sql.window import Window
from databricks.sdk.runtime import spark, udf

from src.utils import data_cleaning
from src.globals import ANNUAL_OUTPUTS_TABLE

from dmlutils import constants
from dmlutils.building_upgrades.upgrades import Upgrade, upgrades_df, get_upgrade_id
from dmlutils.surrogate_model.apply_upgrades import (
    extract_r_value,
    extract_cooling_efficiency,
    extract_heating_efficiency,
    BASIC_ENCLOSURE_INSULATION,
    APPLIANCE_FUEL_COLS,
    GAS_APPLIANCE_INDICATOR_COLS,
)

#  -- constants -- #
SUPPORTED_UPGRADES = [
    Upgrade.BASELINE_2022_1.value,
    Upgrade.BASELINE_2024_2_NO_SETBACK.value,
    Upgrade.BASIC_ENCLOSURE.value,
    Upgrade.MIN_EFF_HP_ELEC_BACKUP.value,
    Upgrade.HIGH_EFF_HP_ELEC_BACKUP.value,
    Upgrade.HP_WATER_HEATER.value,
    Upgrade.WHOLE_HOME_ELECTRIC_MAX_EFF_BASIC_ENCLOSURE.value,
    Upgrade.MED_EFF_HP_HERS_SIZING_NO_SETBACK_2022_1.value,
    Upgrade.MED_EFF_HP_HERS_SIZING_NO_SETBACK_2024_2.value,
    Upgrade.MED_EFF_HP_HERS_SIZING_NO_SETBACK_BASIC_ENCLOSURE.value,
    Upgrade.MED_EFF_HP_HERS_SIZING_NO_SETBACK_LIGHT_TOUCH_AIR_SEALING.value,
    Upgrade.DAIKIN_DUCTLESS_COLD_CLIMATE_HP_HERS_No_SETBACK.value,
    Upgrade.CARRIER_DUCTLESS_PERFORMANCE_COLD_CLIMATE_HP_HERS_No_SETBACK.value,
    Upgrade.YORK_DUCTLESS_COLD_CLIMATE_HP_HERS_No_SETBACK.value,
    Upgrade.DAIKIN_DUCTED_COLD_CLIMATE_HP_HERS_No_SETBACK.value,
]

# mapping of window description to ufactor and shgc (solar heat gain coefficient) pulled from options.tsv
WINDOW_DESCRIPTION_TO_SPEC = spark.createDataFrame(
    [
        ("Double, Clear, Thermal-Break, Air", 0.63, 0.62),
        ("Double, Low-E, H-Gain", 0.29, 0.56),
        ("Double, Low-E, L-Gain", 0.26, 0.31),
        ("Single, Clear, Metal, Exterior Low-E Storm", 0.57, 0.47),
        ("Single, Clear, Non-metal, Exterior Low-E Storm", 0.36, 0.46),
        ("Double, Clear, Metal, Exterior Low-E Storm", 0.49, 0.44),
        ("Double, Clear, Non-metal, Exterior Low-E Storm", 0.28, 0.42),
        ("Double, Clear, Metal, Air", 0.76, 0.67),
        ("Double, Clear, Metal, Air, Exterior Clear Storm", 0.55, 0.51),
        ("Double, Clear, Non-metal, Air", 0.49, 0.56),
        ("Double, Clear, Non-metal, Air, Exterior Clear Storm", 0.34, 0.49),
        ("Double, Low-E, Non-metal, Air, M-Gain", 0.38, 0.44),
        ("Double, Low-E, Non-metal, Air, L-Gain", 0.37, 0.30),
        ("Single, Clear, Metal", 1.16, 0.76),
        ("Single, Clear, Metal, Exterior Clear Storm", 0.67, 0.56),
        ("Single, Clear, Non-metal", 0.84, 0.63),
        ("Single, Clear, Non-metal, Exterior Clear Storm", 0.47, 0.54),
        ("Triple, Low-E, Non-metal, Air, L-Gain", 0.29, 0.26),
        ("Triple, Low-E, Insulated, Argon, H-Gain", 0.18, 0.40),
        ("Triple, Low-E, Insulated, Argon, L-Gain", 0.17, 0.27),
        ("No Windows", 0.84, 0.63),
    ],
    ("windows", "window_ufactor", "window_shgc"),
)

#  -- resstock reading and preprocessing functions  -- #


def clean_building_metadata(raw_resstock_metadata_df: DataFrame) -> DataFrame:
    """
    Rename and remove columns of a ResStock building metadata dataframe

    Can either pass a parquet file or an existing DataFrame.

    Args:
        raw_resstock_metadata_df (DataFrame): DataFrame containing raw ResStock metadata
    Returns:
        building_metadata_cleaned (DataFrame): cleaned ResStock building metadata

    """
    # Read in data and modify pkey name and dtype
    building_metadata = raw_resstock_metadata_df.withColumn("building_id", F.col("bldg_id").cast("int")).drop("bldg_id")

    # rename and remove columns
    building_metadata_cleaned = data_cleaning.edit_columns(
        df=building_metadata,
        remove_substrings_from_columns=["in__"],
        remove_columns_with_substrings=[
            "simulation_control_run",
            "emissions",
            "weight",
            "applicability",
            "upgrade",
            "out__",
            "utility_bill",
            "metadata_index",
        ],
    )

    # Filter to homes of interest: occupied sf homes with modeled fuels and without shared HVAC systems
    filtered_building_metadata = (
        building_metadata_cleaned
        # only single family, mobile home, or multifam with < 5 units
        .where(
            F.col("geometry_building_type_acs").isin(
                [
                    "Single-Family Detached",
                    "Single-Family Attached",
                    "Mobile Home",
                    "2 Unit",
                    "3 or 4 Unit",
                ]
            )
        )
        # other fuels are not modeled in resstock, and we are not including homes without heating due to poor performance
        # and this filter is sufficienct to remove units that have other fuels for any applaince
        .where(~F.col("heating_fuel").isin(["Other Fuel", "None"])).where(F.col("water_heater_fuel") != "Other Fuel")
        # filter out vacant homes
        .where(F.col("vacancy_status") == "Occupied")
        # filter out homes with shared HVAC or water heating systems
        .where((F.col("hvac_has_shared_system") == "None") & (F.col("water_heater_in_unit") == "Yes"))
    )

    return filtered_building_metadata


#  -- feature transformation udfs -- #
@udf(returnType=DoubleType())
def extract_percentage(value: str) -> float:
    """Extract percentage from string and divide by 100

    >>> extract_percentage('100% Conditioned')
    1.0
    >>> extract_percentage('<10% Conditioned')
    0.1
    >>> extract_percentage('None')
    0.0
    >>> extract_percentage('10% Leakage, Uninsulated')
    0.1
    """
    if value == "None":
        return 0.0
    match = re.match(r"^<?(\d+)%", value)
    try:
        return (match and float(match.group(1))) / 100.0
    except ValueError:
        raise ValueError(f"Cannot extract percentage from: f{value}")


@udf(returnType=IntegerType())
def extract_vintage(vintage: str) -> int:
    """return the midpoint of the vintage decade bin, with '<1940' as 1930
    >>> extract_vintage('<1940')
    1930
    >>> extract_vintage('1960s')
    1965
    """
    vintage = vintage.strip()
    if vintage.startswith("<"):  # '<1940' bin in resstock
        return 1930
    return int(vintage[:4]) + 5


@udf(returnType=IntegerType())
def extract_r_valueUDF(construction_type: str, set_none_to_inf: bool = False) -> int:
    return extract_r_value(construction_type, set_none_to_inf)


extract_cooling_efficiencyUDF = udf(lambda x: extract_cooling_efficiency(x), DoubleType())

extract_heating_efficiencyUDF = udf(lambda x: extract_heating_efficiency(x), DoubleType())


@udf(returnType=DoubleType())
def extract_temp(temperature_string) -> float:
    """Convert string Fahrenheit degrees to float F

    >>> extract_temp('70F')
    70.0
    >>> extract_temp('-3F')
    -3.0
    """
    if not re.match(r"\d+F", temperature_string):
        raise ValueError(f"Unrecognized temperature format: {temperature_string}")
    return float(temperature_string.strip().lower()[:-1])


@udf(returnType=DoubleType())
def extract_mean_wwr(value: str) -> int:
    """
    Return the average window to wall ratio (WWR) for front, back, left, and right walls.
    >>> extract_mean_wwr('F9 B9 L9 R9')
    9.0
    >>> extract_mean_wwr('F12 B12 L12 R12')
    12.0
    >>> extract_mean_wwr('Uninsulated')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Unrecognized format of window area: ...
    """
    try:
        wwr_list = [int(side[1:]) for side in value.split()]
        return sum(wwr_list) / len(wwr_list)
    except ValueError:
        raise ValueError(f"Unrecognized format of window area: {value}")


@udf(returnType=DoubleType())
def extract_energy_factor(ef_string: str) -> int:
    """
    >>> extract_energy_factor("EF 10.2, 100% Usage")
    10.2
    >>> extract_energy_factor("EF 6.7")
    6.7
    >>> extract_energy_factor("None")
    99.
    """
    if "None" in ef_string:
        return 99.0
    return float(ef_string.split(",")[0][3:])


#  -- water heater transformation udf and helper functions -- #
@udf(IntegerType())
def get_water_heater_capacity_ashrae(n_bedrooms: int, n_bathrooms: float, is_electric: bool) -> int:
    """
    Calculates the recommended water heater capacity in gallons based on
    the number of bedrooms, bathrooms, and whether the heater is electric.
    Source: https://www.nrel.gov/docs/fy10osti/47246.pdf
    Table 8. Benchmark Domestic Hot Water Storage and Burner Capacity (ASHRAE 1999)

    >>> get_water_heater_capacity_ashrae(3, 4, False)
    40
    >>> get_water_heater_capacity_ashrae(1, 1.5, True)
    20
    >> get_water_heater_capacity_ashrae(6, 5, True)
    80
    """
    match n_bedrooms:
        case 1:
            return 20
        case 2:
            if n_bathrooms < 2:
                return 30
            elif n_bathrooms < 3:
                if is_electric:
                    return 40
                else:
                    return 30
            else:
                if is_electric:
                    return 50
                else:
                    return 40
        case 3:
            if n_bathrooms < 2:
                if is_electric:
                    return 40
                else:
                    return 30
            else:
                if is_electric:
                    return 50
                else:
                    return 40
        case 4:
            if n_bathrooms < 3:
                if is_electric:
                    return 50
                else:
                    return 40
            else:
                if is_electric:
                    return 66
                else:
                    return 50
        case 5:
            if is_electric:
                return 66
            else:
                return 50
        case 6:
            if is_electric:
                return 80
            else:
                return 50


# Define the schema for the output struct
wh_schema = StructType(
    [
        StructField("water_heater_type", StringType(), True),  # "Storage", "Heat Pump", or "Instantaneous"
        StructField("water_heater_tank_volume_gal", IntegerType(), True),  # Capacity of the tank in gallons
        StructField("water_heater_efficiency_ef", DoubleType(), True),  # Efficiency Factor (EF)
        StructField("water_heater_recovery_efficiency_ef", DoubleType(), True),  # Recovery Efficiency Factor (EF)
    ]
)


# pulled from options.tsv
@udf(wh_schema)
def get_water_heater_specs(name: str) -> StructType:
    """
    Parses the name of a water heater to extract and compute its specifications based on the ResStock options.tsv
    >>> lookup_water_heater_specs("Natural Gas Tankless")
        {'water_heater_type': 'Instantaneous',
        'water_heater_tank_volume_gal': 0,
        'water_heater_efficiency_ef': 0.82,
        'water_heater_recovery_efficiency_ef': 1.0}
    >>> lookup_water_heater_specs("FIXME Fuel Oil Indirect")
        {'water_heater_type': 'Storage',
        'water_heater_tank_volume_gal': 0,
        'water_heater_efficiency_ef': 0.62,
        'water_heater_recovery_efficiency_ef': 0.78}
    >>> lookup_water_heater_specs("Electric Heat Pump, 50 gal, 3.45 UEF")
        {'water_heater_type': 'Heat Pump',
        'water_heater_tank_volume_gal': 50,
        'water_heater_efficiency_ef': 3.57,
        'water_heater_recovery_efficiency_ef': 1.0}

    """
    # initialize dict of outputs with defaults
    specs = {
        "water_heater_type": "Storage",  # default for non-tankless and non-hpwhs
        "water_heater_tank_volume_gal": None,  # no default set
        "water_heater_efficiency_ef": None,  # no default set
        "water_heater_recovery_efficiency_ef": 1.0,  # default for electric and tankless
    }

    # Split the name into constituent parts
    def split_wh_name(name):
        "Splits name into 3 elements on a comma delimiter, returning empty string for elements that are not present"
        parts = name.split(", ")
        # Extend the list with None to ensure it has at least 3 elements
        parts += [""] * (3 - len(parts))
        return parts[0], parts[1], parts[2]

    fuel_and_type, tank_volume, efficiency = split_wh_name(name)

    # Extract the water heater type
    if "Tankless" in fuel_and_type:
        specs["water_heater_type"] = "Instantaneous"
        specs["water_heater_tank_volume_gal"] = 0
    elif fuel_and_type == "Electric Heat Pump":
        specs["water_heater_type"] = "Heat Pump"

    # Set the efficiency and recovery efficiency
    match fuel_and_type:
        case "Natural Gas Standard" | "Propane Standard":
            specs["water_heater_efficiency_ef"] = 0.59
            specs["water_heater_recovery_efficiency_ef"] = 0.76
        case "Fuel Oil Standard" | "FIXME Fuel Oil Indirect":
            specs["water_heater_efficiency_ef"] = 0.62
            specs["water_heater_recovery_efficiency_ef"] = 0.78
        case "Natural Gas Premium" | "Propane Premium":
            specs["water_heater_efficiency_ef"] = 0.67
            specs["water_heater_recovery_efficiency_ef"] = 0.78
        case "Fuel Oil Premium":
            specs["water_heater_efficiency_ef"] = 0.68
            specs["water_heater_recovery_efficiency_ef"] = 0.9
        case "Natural Gas Tankless" | "Propane Tankless":
            specs["water_heater_efficiency_ef"] = 0.82
        case "Electric Standard":
            specs["water_heater_efficiency_ef"] = 0.92
        case "Electric Premium":
            specs["water_heater_efficiency_ef"] = 0.95
        case "Electric Tankless":
            specs["water_heater_efficiency_ef"] = 0.99
        case "Electric Heat Pump":
            specs["water_heater_efficiency_ef"] = 2.3
    # If Uniform Energy Factor (UEF) is specified (should only ever be specified for HPWH)
    # use this for efficiency and convert to Energy Factor (EF).
    uef_efficiency_match = re.search(r"(\d+\.\d+)\s*UEF", efficiency)
    if uef_efficiency_match:
        uef = float(uef_efficiency_match.group(1))
        # Convert UEF to EF for HPWH. Source: #https://www.resnet.us/wp-content/uploads/RESNET-EF-Calculator-2017.xlsx
        # Note that other water heater types require a sligntly different equation.
        specs["water_heater_efficiency_ef"] = round(1.2101 * uef - 0.6052, 3)

    # Extract the tank volume if specified
    tank_volume_match = re.search(r"(\d+)\s*gal", tank_volume)
    if tank_volume_match:
        specs["water_heater_tank_volume_gal"] = int(tank_volume_match.group(1))

    return specs


def add_water_heater_features(df):
    return (
        df.withColumn("wh_struct", get_water_heater_specs(F.col("water_heater_efficiency")))
        .withColumn("water_heater_type", F.col("wh_struct.water_heater_type"))
        .withColumn(
            "water_heater_tank_volume_gal",
            F.col("wh_struct.water_heater_tank_volume_gal"),
        )
        .withColumn("water_heater_efficiency_ef", F.col("wh_struct.water_heater_efficiency_ef"))
        .withColumn(
            "water_heater_recovery_efficiency_ef",
            F.col("wh_struct.water_heater_recovery_efficiency_ef"),
        )
        .drop("wh_struct")
    )


#  -- various mapping expressions -- #
def make_map_type_from_dict(mapping: Dict) -> Column:
    """
    Create a MapType mapping from a dict to pass to Pyspark
    https://stackoverflow.com/a/42983199
    """
    return F.create_map([F.lit(x) for x in chain(*mapping.items())])


yes_no_mapping = make_map_type_from_dict({"Yes": True, "No": False})

low_medium_high_mapping = make_map_type_from_dict({"Low": 1, "Medium": 2, "High": 3})

orientation_degrees_mapping = make_map_type_from_dict(
    {
        "North": 0,
        "Northeast": 45,
        "East": 90,
        "Southeast": 135,
        "South": 180,
        "Southwest": 225,
        "West": 270,
        "Northwest": 315,
    }
)

# https://en.wikipedia.org/wiki/Luminous_efficacy
luminous_efficiency_mapping = make_map_type_from_dict(
    {
        "100% CFL": 0.12,  # 8-15%
        "100% Incandescent": 0.02,  # 1.2-2.6%
        "100% LED": 0.15,  # 11-30%
    }
)


#  -- function to apply all the baseline transformations -- #
def transform_building_features(building_metadata_table_name) -> DataFrame:
    """
    Read and transform building metadata into features.

    Args:
        building_metadata_table_name : name of building metadata delta table

    Returns:
        Dataframe: dataframe of building metadata features
    """
    building_metadata_features = (
        spark.read.table(building_metadata_table_name)
        # -- structure transformations -- #
        .withColumn("n_bedrooms", F.col("bedrooms").cast("int"))
        .withColumn("n_bathrooms", F.col("n_bedrooms") / 2 + 0.5)  # based on docs
        .withColumn("orientation_degrees", orientation_degrees_mapping[F.col("orientation")])
        .withColumn(
            "garage_size_n_car",
            F.coalesce(F.split(F.col("geometry_garage"), " ")[0].cast("int"), F.lit(0)),
        )
        .withColumn(
            "geometry_wall_exterior_finish",
            F.regexp_replace("geometry_wall_exterior_finish", "Shingle, ", "Shingle-"),
        )
        .withColumn("exterior_wall_material", F.split("geometry_wall_exterior_finish", ", ")[0])
        .withColumn(
            "exterior_wall_color",
            F.coalesce(F.split("geometry_wall_exterior_finish", ", ")[1], F.lit("None")),
        )
        .withColumn("window_wall_ratio", extract_mean_wwr(F.col("window_areas")))
        .join(WINDOW_DESCRIPTION_TO_SPEC, on="windows")
        # -- heating tranformations -- #
        .withColumn(
            "heating_appliance_type",
            F.expr("replace(hvac_heating_type_and_fuel, heating_fuel, '')"),
        )
        .withColumn(
            "heating_appliance_type",
            F.when(F.col("heating_appliance_type").contains("Wall Furnace"), "Wall Furnace")
            .when(F.col("heating_appliance_type").contains("Furnace"), "Furnace")
            .when(F.col("heating_appliance_type").contains("Boiler"), "Boiler")
            .when(F.col("heating_appliance_type") == "", "None")
            .otherwise(F.trim(F.col("heating_appliance_type"))),
        )
        .withColumn(
            "heat_pump_sizing_methodology",
            F.when(F.col("heating_appliance_type") == "ASHP", F.lit("ACCA")).otherwise("None"),
        )
        .withColumn(
            "heating_efficiency_nominal_percentage",
            extract_heating_efficiencyUDF(F.col("hvac_heating_efficiency")),
        )
        .withColumn(
            "heating_setpoint_degrees_f",
            extract_temp(F.col("heating_setpoint")),
        )
        .withColumn(
            "heating_setpoint_offset_magnitude_degrees_f",
            extract_temp(F.col("heating_setpoint_offset_magnitude")),
        )
        .withColumn(  # note there are cases where hvac_has_ducts = True, but heating system is still ductless
            "has_ducted_heating",
            F.col("hvac_heating_type").isin("Ducted Heat Pump", "Ducted Heating"),
        )
        # -- cooling tranformations -- #
        .withColumn("ac_type", F.split(F.col("hvac_cooling_efficiency"), ",")[0])
        .withColumn(
            "cooled_space_percentage",
            extract_percentage(F.col("hvac_cooling_partial_space_conditioning")),
        )
        .withColumn(
            "cooling_efficiency_eer_str",
            F.when(
                F.col("hvac_cooling_efficiency") == "Heat Pump",
                F.col("hvac_heating_efficiency"),
            ).otherwise(F.col("hvac_cooling_efficiency")),
        )
        .withColumn(
            "cooling_efficiency_eer",
            extract_cooling_efficiencyUDF(F.col("cooling_efficiency_eer_str")),
        )
        .withColumn(
            "cooling_setpoint_degrees_f",
            extract_temp(F.col("cooling_setpoint")),
        )
        .withColumn(
            "cooling_setpoint_offset_magnitude_degrees_f",
            extract_temp(F.col("cooling_setpoint_offset_magnitude")),
        )
        .withColumn(  # note there are cases where hvac_has_ducts = True, but cooling system is still ductless
            "has_ducted_cooling",
            F.col("hvac_cooling_type").isin(["Central AC", "Ducted Heat Pump"]),
        )
        # -- water heating tranformations -- #
        .transform(add_water_heater_features)
        .withColumn(
            "water_heater_tank_volume_gal_ashrae",
            get_water_heater_capacity_ashrae(
                F.col("n_bedrooms"),
                F.col("n_bathrooms"),
                F.col("water_heater_fuel") == "Electricity",
            ),
        )
        .withColumn(
            "water_heater_tank_volume_gal",
            F.coalesce(
                F.col("water_heater_tank_volume_gal"),
                F.col("water_heater_tank_volume_gal_ashrae"),
            ),
        )
        .withColumn("has_water_heater_in_unit", yes_no_mapping[F.col("water_heater_in_unit")])
        # -- duct/infiltration tranformations -- #
        .withColumn("has_ducts", yes_no_mapping[F.col("hvac_has_ducts")])
        .withColumn("duct_insulation_r_value", extract_r_valueUDF(F.col("duct_leakage_and_insulation"), F.lit(True)))
        .withColumn("duct_leakage_percentage", extract_percentage(F.col("duct_leakage_and_insulation")))
        .withColumn("infiltration_ach50", F.split(F.col("infiltration"), " ")[0].cast("int"))
        # -- insulation tranformations -- #
        .withColumn("wall_material", F.split(F.col("insulation_wall"), ",")[0])
        .withColumn("insulation_wall_r_value", extract_r_valueUDF(F.col("insulation_wall")))
        .withColumn(
            "insulation_foundation_wall_r_value",
            extract_r_valueUDF(F.col("insulation_foundation_wall")),
        )
        .withColumn("insulation_slab_r_value", extract_r_valueUDF(F.col("insulation_slab")))
        .withColumn(
            "insulation_rim_joist_r_value",
            extract_r_valueUDF(F.col("insulation_rim_joist")),
        )
        .withColumn("insulation_floor_r_value", extract_r_valueUDF(F.col("insulation_floor")))
        .withColumn("insulation_ceiling_r_value", extract_r_valueUDF(F.col("insulation_ceiling")))
        .withColumn("insulation_roof_r_value", extract_r_valueUDF(F.col("insulation_roof")))
        #  -- building type transformations -- #
        .withColumn(
            "is_attached",
            ~F.col("geometry_building_type_acs").isin(["Single-Family Detached", "Mobile Home"]),
        )
        .withColumn(
            "is_mobile_home",
            F.col("geometry_building_type_acs") == "Mobile Home",
        )
        # prep for building unit transform so that coalesce will work in next step
        # value for *_sfa will be "None" for non sfa buildings and values for *_mf will be null for non-mf
        .replace(
            "None",
            None,
            subset=[
                "geometry_building_number_units_sfa",
                "geometry_building_number_units_mf",
            ],
        )
        # sf detatched and mobile homes will be Null for both and should get mapped to 1 unit
        .withColumn(
            "n_building_units",
            F.coalesce(
                F.col("geometry_building_number_units_sfa"),
                F.col("geometry_building_number_units_mf"),
                F.lit("1"),
            ).cast("int"),
        )
        .withColumn(
            "is_middle_unit",
            (F.col("geometry_building_horizontal_location_sfa") == "Middle")
            | (F.col("geometry_building_horizontal_location_mf") == "Middle"),
        )
        # -- other appliances -- #
        .withColumn("has_ceiling_fan", F.col("ceiling_fan") != "None")
        .withColumn("clothes_dryer_fuel", F.col("clothes_dryer"))
        .withColumn("has_induction_range", F.col("cooking_range") == "Electric Induction")
        .withColumn(
            "cooking_range_fuel",
            F.regexp_replace("cooking_range", " Induction| Resistance", ""),
        )
        .withColumn(
            "dishwasher_efficiency_kwh",
            F.coalesce(F.split(F.col("dishwasher"), " ")[0].cast("int"), F.lit(9999)),
        )
        .withColumn("lighting_efficiency", luminous_efficiency_mapping[F.col("lighting")])
        .withColumn(
            "refrigerator_extra_efficiency_ef",
            extract_energy_factor(F.col("misc_extra_refrigerator")),
        )
        .withColumn("has_standalone_freezer", F.col("misc_freezer") != "None")
        .withColumn("has_gas_fireplace", F.col("misc_gas_fireplace") != "None")
        .withColumn("has_gas_grill", F.col("misc_gas_grill") != "None")
        .withColumn("has_gas_lighting", F.col("misc_gas_lighting") != "None")
        .withColumnRenamed("misc_hot_tub_spa", "hot_tub_spa_fuel")
        .withColumnRenamed("misc_pool_heater", "pool_heater_fuel")
        .withColumn("has_well_pump", F.col("misc_well_pump") != "None")
        .withColumn("refrigerator_efficiency_ef", extract_energy_factor(F.col("refrigerator")))
        .withColumn("plug_load_percentage", extract_percentage(F.col("plug_loads")))
        .withColumn("usage_level_appliances", low_medium_high_mapping[F.col("usage_level")])
        # -- misc transformations -- #
        .withColumn(
            "climate_zone_temp",
            F.substring("ashrae_iecc_climate_zone_2004", 1, 1).astype("int"),
        )
        .withColumn("climate_zone_moisture", F.substring("ashrae_iecc_climate_zone_2004", 2, 1))
        .withColumn(
            "neighbor_distance_ft",
            F.when(F.col("neighbors") == "Left/Right at 15ft", 15.0)
            .when(F.col("neighbors") == "None", 9999.0)
            .otherwise(F.col("neighbors").cast("double")),
        )
        .withColumn(
            "n_occupants",
            F.when(F.col("occupants") == "10+", 11).otherwise(F.col("occupants").cast("int")),
        )
        .withColumn("vintage", extract_vintage(F.col("vintage")))
        # align names for electricity and natural gas across applainces
        .replace(
            {
                "Gas": "Natural Gas",
                "Electric": "Electricity",
            },
            subset=APPLIANCE_FUEL_COLS,
        )
        # subset to all possible features of interest
        .select(
            # primary key
            "building_id",
            "building_set",
            # foreign key
            "weather_file_city",
            # structure
            "n_bedrooms",
            "n_bathrooms",
            F.col("geometry_attic_type").alias("attic_type"),
            F.col("sqft").cast("double"),
            F.col("geometry_foundation_type").alias("foundation_type"),
            "garage_size_n_car",
            F.col("geometry_stories").cast("int").alias("n_stories"),
            "orientation_degrees",
            "roof_material",
            "window_wall_ratio",
            "window_ufactor",
            "window_shgc",
            # heating
            "heating_fuel",
            "heating_appliance_type",
            "heat_pump_sizing_methodology",
            "has_ducted_heating",
            "heating_efficiency_nominal_percentage",
            "heating_setpoint_degrees_f",
            "heating_setpoint_offset_magnitude_degrees_f",
            "hvac_heating_efficiency",  # only used for applying upgrades, gets dropped later
            # cooling
            "ac_type",
            "cooled_space_percentage",
            "cooling_efficiency_eer",
            "cooling_setpoint_degrees_f",
            "cooling_setpoint_offset_magnitude_degrees_f",
            "has_ducted_cooling",
            # water heater
            "water_heater_efficiency",  # only used for applying upgrades, gets dropped later
            "water_heater_fuel",
            "water_heater_type",
            "water_heater_tank_volume_gal",
            "water_heater_efficiency_ef",
            "water_heater_recovery_efficiency_ef",
            "has_water_heater_in_unit",
            "water_heater_location",
            # ducts
            "duct_leakage_and_insulation",  # only used for applying upgrades, gets dropped later
            "has_ducts",
            "duct_location",
            "duct_insulation_r_value",
            "duct_leakage_percentage",
            "infiltration_ach50",
            # insulalation
            "insulation_wall",  # only used for applying upgrades, gets dropped later
            "wall_material",
            "insulation_wall_r_value",
            "insulation_foundation_wall_r_value",
            "insulation_slab_r_value",
            "insulation_rim_joist_r_value",
            "insulation_floor_r_value",
            "insulation_roof_r_value",
            "insulation_ceiling_r_value",
            # building type
            "is_attached",
            "is_mobile_home",
            "n_building_units",
            "is_middle_unit",
            F.col("geometry_building_level_mf").alias("unit_level_in_building"),
            # other appliances
            "has_ceiling_fan",
            "clothes_dryer_fuel",
            F.col("clothes_washer").alias("clothes_washer_efficiency"),
            "cooking_range_fuel",
            "has_induction_range",
            "dishwasher_efficiency_kwh",
            "lighting_efficiency",
            "refrigerator_extra_efficiency_ef",
            "has_standalone_freezer",
            "has_gas_fireplace",
            "has_gas_grill",
            "has_gas_lighting",
            "has_well_pump",
            "hot_tub_spa_fuel",
            "pool_heater_fuel",
            "refrigerator_efficiency_ef",
            "plug_load_percentage",
            "usage_level_appliances",
            # misc
            "climate_zone_temp",
            "climate_zone_moisture",
            "neighbor_distance_ft",
            "n_occupants",
            "vintage",
        )
    )
    return building_metadata_features


#  -- functions to apply upgrade transformations -- #

# Mapping of climate zone temperature  -> threshold, insulation
# where climate zone temperature is the first character in the ASHRAE IECC climate zone
# ('1', 13, 30) means units in climate zones 1A (1-anything) with R13 insulation or less are upgraded to R30
BASIC_ENCLOSURE_INSULATION_SPARK = spark.createDataFrame(BASIC_ENCLOSURE_INSULATION)

# Define mapping of "hvac_heating_efficiency" to parameters of performance curve based on energy plus options.tsv in each upgrade config
# This defines the capacity and cop at min and max speeds for 3 outdoor temperatures
# NOTE: schema will need to further identify the heat pump if ever there are hvac_heating_efficiency's that map to multiple performance curves
performance_curve_parameter_schema = StructType(
    [
        StructField("hvac_heating_efficiency", StringType(), True),
        StructField("min_capacity_retention_47f", DoubleType(), True),
        StructField("min_capacity_retention_17f", DoubleType(), True),
        StructField("min_capacity_retention_5f", DoubleType(), True),
        StructField("max_capacity_retention_47f", DoubleType(), True),
        StructField("max_capacity_retention_17f", DoubleType(), True),
        StructField("max_capacity_retention_5f", DoubleType(), True),
        StructField("min_cop_47f", DoubleType(), True),
        StructField("min_cop_17f", DoubleType(), True),
        StructField("min_cop_5f", DoubleType(), True),
        StructField("max_cop_47f", DoubleType(), True),
        StructField("max_cop_17f", DoubleType(), True),
        StructField("max_cop_5f", DoubleType(), True),
    ]
)
# TODO: add in performance curve params for NREL baseline HPs and for ductless NREL HPs
performance_curve_parameter_data = [
    ("ASHP, SEER 15, 9 HSPF", 0.98, 0.62, 0.50, 1.01, 0.64, 0.52, 3.80, 2.79, 2.41, 4.34, 3.23, 2.78),
    ("MSHP, SEER 24, 13 HSPF", 0.97, 0.73, 0.63, 1.03, 0.79, 0.69, 5.72, 3.66, 3.26, 8.27, 5.93, 5.31),
    ("ASHP, SEER 18, 10 HSPF", 1.01, 0.60, 0.44, 1.04, 0.63, 0.47, 4.23, 2.57, 1.98, 5.83, 3.44, 2.85),
    ("Daikin MSHP SZ, SEER 22.05, 10.64 HSPF", 0.24, 0.15, 0.12, 1.13, 0.98, 0.79, 5.76, 5.55, 5.47, 4.25, 2.63, 1.8),
    ("Daikin MSHP MZ, SEER 22.05, 11.2 HSPF", 0.32, 0.22, 0.18, 1.33, 1.08, 1.0, 4.95, 4.85, 4.96, 2.61, 1.99, 1.80),
    ("Carrier MSHP SZ, SEER 24.255, 13.104 HSPF", 0.4, 0.24, 0.26, 1.23, 0.77, 0.78, 5.58, 2.37, 2.0, 2.87, 2.33, 2.54),
    ("Carrier MSHP MZ, SEER 26.25, 10.64 HSPF", 0.34, 0.24, 0.22, 1.16, 0.76, 0.72, 4.92, 2.93, 2.43, 3.54, 2.3, 2.0),
    ("York MSHP SZ, SEER 23.625, 11.21 HSPF", 0.33, 0.42, 0.21, 1.04, 1.0, 0.8, 4.7, 1.95, 1.96, 2.4, 1.86, 1.80),
    ("York MSHP MZ, SEER 25.36, 11.2 HSPF", 0.44, 0.42, 0.12, 1.55, 1.01, 0.95, 2.3, 1.45, 0.3, 2.9, 2.4, 1.8),
    ("Daikin 7 series ASHP, SEER 20, 10.35 HSPF", 0.25, 0.41, 0.34, 1.0, 0.94, 0.73, 4.32, 2.75, 2.36, 3.3, 2.32, 2.0),
]

HEAT_PUMP_PERFORMANCE_CURVE_DF = spark.createDataFrame(
    performance_curve_parameter_data, schema=performance_curve_parameter_schema
)


def fill_null_with_column(df, source_column, columns_to_fill):
    """
    Fills null values in specified columns with the value from a source column

    Args:
        df: Input DataFrame
        source_column: Column name to use as the fill value
        columns_to_fill: List of column names to fill when null

    Returns:
        DataFrame with null values filled
    """
    # Create a dictionary of column expressions
    column_exprs = {
        column: F.when(F.col(column).isNull(), F.col(source_column)).otherwise(F.col(column))
        for column in columns_to_fill
    }

    # Apply all transformations at once
    return df.select(
        *[F.col(c) for c in df.columns if c not in columns_to_fill],  # keep all other columns as-is
        *[column_exprs[c].alias(c) for c in columns_to_fill],  # apply our transformations
    )


def remove_setbacks(building_features: DataFrame) -> DataFrame:
    """
    Remove setbacks by setting the cooling and heating setpoint offset magnitudes to zero.

    Args:
        building_features (DataFrame): The baseline building features.

    Returns:
        DataFrame: The building features DataFrame with setbacks removed.
    """
    return building_features.withColumn("cooling_setpoint_offset_magnitude_degrees_f", F.lit(0.0)).withColumn(
        "heating_setpoint_offset_magnitude_degrees_f", F.lit(0.0)
    )


def update_hp_ducted_ductless(
    building_features: DataFrame,
    ducted_efficiency: str,
    non_ducted_efficiency: str,
) -> DataFrame:
    """
    Upgrade the baseline building features for heating and cooling efficiencies and ducted heating/cooling status
    with a ducted heat pump if the home has ducts and a ductless heat pump if the home does not have ducts.
    TODO: make the specified upgrade thresholds  (<=SEER 15; <=HSPF 8.5) explicit here. In 2024.2 there are now
    baseline hps that exceed these thresholds (MSHP, SEER 29.3, 14 HSPF).

    Args:
        building_features (DataFrame): The baseline building features.
        ducted_efficiency (str): The efficiency of the ducted heat pump.
        non_ducted_efficiency (str): The efficiency of the ductless heat pump.

    Returns:
        DataFrame: The upgraded building features DataFrame with the heat pump logic applied.
    """
    return (
        building_features.withColumn(
            "heating_efficiency_nominal_percentage",
            F.when(F.col("has_ducts"), extract_heating_efficiencyUDF(F.lit(ducted_efficiency))).otherwise(
                extract_heating_efficiencyUDF(F.lit(non_ducted_efficiency))
            ),
        )
        .withColumn(
            "cooling_efficiency_eer",
            F.when(
                F.col("has_ducts"),
                extract_cooling_efficiencyUDF(F.lit(ducted_efficiency)),
            ).otherwise(extract_cooling_efficiencyUDF(F.lit(non_ducted_efficiency))),
        )
        .withColumn("has_ducted_heating", F.col("has_ducts"))
        .withColumn("has_ducted_cooling", F.col("has_ducts"))
        # Add column to join to performance curve metrics on where performance metrics are currently based on ducted ducted
        # TODO: generalize this to work for ducted or ductless when we have those params
        .withColumn("hvac_heating_efficiency", F.lit(ducted_efficiency))
    )


def get_hp_efficiencies_single_multi_zone_ductless(
    building_features: DataFrame, single_zone_efficiency: str, multi_zone_efficiency: str
) -> DataFrame:
    """
    Upgrade the baseline building features for heating and cooling efficiencies with a ductless heat pump.
    Applies single-zone or multi-zone efficiency based on the size of the home and joins performance curve parameters.

    Args:
        building_features (DataFrame): The baseline building features.
        single_zone_efficiency (str): The efficiency of the single-zone ductless heat pump.
        multi_zone_efficiency (str): The efficiency of the multi-zone ductless heat pump.

    Returns:
        DataFrame: The upgraded building features DataFrame with the ductless heat pump logic applied.
    """

    def single_zone_is_applicable():
        """
        Define the condition for home size logic reuse.
        """
        return (
            (F.col("n_bedrooms") == 1) | (F.col("sqft") <= 749) | ((F.col("n_bedrooms") == 2) & (F.col("sqft") <= 999))
        )

    # Package apply logic (return rows unchanged if they don't match the conditions)
    building_features = building_features.withColumn(
        "upgrade_is_applicable",
        (
            (F.col("heating_fuel") != "None")
            & (
                (
                    F.col("heating_efficiency_nominal_percentage")
                    < extract_heating_efficiency("ASHP, SEER 25, 12.7 HSPF")
                )
                | (F.col("cooling_efficiency_eer") < extract_cooling_efficiency("ASHP, SEER 25, 12.7 HSPF"))
            )
        ),
    )
    # add column on with string on hp efficiency which we can extract heating and cooling efficiencies
    # and performance curve metrics
    building_features = building_features.withColumn(
        "hvac_heating_efficiency",
        F.when(
            F.col("upgrade_is_applicable"),
            F.when(single_zone_is_applicable(), single_zone_efficiency).otherwise(multi_zone_efficiency),
        ).otherwise(F.lit("")),
    )

    # extract heating and cooling efficiencies
    building_features = building_features.withColumn(
        "heating_efficiency_nominal_percentage",
        F.when(
            F.col("upgrade_is_applicable"), extract_heating_efficiencyUDF(F.col("hvac_heating_efficiency"))
        ).otherwise(F.col("heating_efficiency_nominal_percentage")),
    ).withColumn(
        "cooling_efficiency_eer",
        F.when(
            F.col("upgrade_is_applicable"), extract_cooling_efficiencyUDF(F.col("hvac_heating_efficiency"))
        ).otherwise(F.col("cooling_efficiency_eer")),
    )

    # heating and cooling is now ductless
    building_features = building_features.withColumn(
        "has_ducted_heating", F.when(F.col("upgrade_is_applicable"), False).otherwise(F.col("has_ducted_heating"))
    ).withColumn(
        "has_ducted_cooling", F.when(F.col("upgrade_is_applicable"), False).otherwise(F.col("has_ducted_cooling"))
    )

    return building_features.drop("upgrade_is_applicable")


def upgrade_to_hp_general(
    building_features: DataFrame,
    heat_pump_sizing_methodology: str = "ACCA",
) -> DataFrame:
    """
    Upgrade the baseline building features to an air source heat pump (ASHP) with logic common to all heat pump upgrades.

    Args:
        building_features (DataFrame): The building features to upgrade.
        heat_pump_sizing_methodology (str, optional) : the name of the heat pump sizing methodology. Defaults to "ACCA".

    Returns:
        DataFrame: The upgraded building features DataFrame with the heat pump.
    """
    return (
        building_features.withColumn("heating_appliance_type", F.lit("ASHP"))
        .withColumn("heating_fuel", F.lit("Electricity"))
        .withColumn("ac_type", F.lit("Heat Pump"))
        .withColumn("cooled_space_percentage", F.lit(1.0))
        .withColumn("heat_pump_sizing_methodology", F.lit(heat_pump_sizing_methodology))
    )


def apply_upgrades(baseline_building_features: DataFrame, upgrade_id: int) -> DataFrame:
    """
    Modify building features to reflect the upgrade.
    https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock
         /2022/EUSS_ResRound1_Technical_Documentation.pdf
    In case of contradictions, consult: https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_10k.yml.
    For RAStock upgrades, consult the config.json and options.tsv in the simulation folder.

    Args:
          baseline_building_features: (DataFrame) building features coming from metadata.
          upgrade_id: (int)

    Returns:
          DataFrame: building_features, augmented to reflect the upgrade.
    """
    # TODO: update this function to use enum names instead of upgrade ids
    # Raise an error if an unsupported upgrade ID is provided
    if upgrade_id not in map(get_upgrade_id, SUPPORTED_UPGRADES):
        raise ValueError(f"Upgrade id={upgrade_id} is not yet supported")

    upgrade_building_features = baseline_building_features.withColumn(
        "upgrade_id", F.lit(upgrade_id).cast("double")
    ).withColumn("has_heat_pump_dryer", F.lit(False))

    if upgrade_id == 0:  # baseline: return as is
        pass

    if upgrade_id == 0.01:  # baseline with no setbacks
        upgrade_building_features = remove_setbacks(upgrade_building_features)

    if upgrade_id in [1, 9, 13.01]:  # basic enclosure
        upgrade_building_features = (
            upgrade_building_features
            # Upgrade insulation of ceiling
            # Map the climate zone number to the insulation params for the upgrade
            .join(BASIC_ENCLOSURE_INSULATION_SPARK, on="climate_zone_temp")
            # Attic floor insulation if current insulation is below threshold
            .withColumn(
                "insulation_ceiling_r_value",
                F.when(
                    (F.col("attic_type") == "Vented Attic")  # TODO: check if new "Unvented Attic" also counts here
                    & (F.col("insulation_ceiling_r_value") <= F.col("existing_insulation_max_threshold")),
                    F.col("insulation_upgrade"),
                ).otherwise(F.col("insulation_ceiling_r_value")),
            )
            .drop("insulation_upgrade", "existing_insulation_max_threshold")
            # Air leakage reduction if high levels of infiltration
            .withColumn(
                "infiltration_ach50",
                F.when(F.col("infiltration_ach50") >= 15, F.col("infiltration_ach50") * 0.7).otherwise(
                    F.col("infiltration_ach50")
                ),
            )
            # Duct sealing: update duct leakage rate to at most 10% and insulation to at least R-8,
            # with the exeption of "0% or 30% Leakage, Uninsulated" which does not get upgrade applied
            .withColumn(
                "duct_leakage_percentage",
                F.when(
                    (
                        (F.col("has_ducts"))
                        & ~(
                            F.col("duct_leakage_and_insulation").isin(
                                ["0% Leakage to Outside, Uninsulated", "30% Leakage to Outside, Uninsulated"]
                            )
                        )
                    ),
                    F.least(F.col("duct_leakage_percentage"), F.lit(0.1)),
                ).otherwise(F.col("duct_leakage_percentage")),
            )
            .withColumn(
                "duct_insulation_r_value",
                F.when(
                    (
                        (F.col("has_ducts"))
                        & ~(
                            F.col("duct_leakage_and_insulation").isin(
                                ["0% Leakage to Outside, Uninsulated", "30% Leakage to Outside, Uninsulated"]
                            )
                        )
                    ),
                    F.greatest(F.col("duct_insulation_r_value"), F.lit(8.0)),
                ).otherwise(F.col("duct_insulation_r_value")),
            )
            # Drill-and-fill wall insulation if the wall type is uninsulated
            .withColumn(
                "insulation_wall_r_value",
                F.when(
                    F.col("insulation_wall") == "Wood Stud, Uninsulated",
                    extract_r_valueUDF(F.lit("Wood Stud, R-13")),
                ).otherwise(F.col("insulation_wall_r_value")),
            )
        )

    if upgrade_id == 13.02:  # light touch air sealing
        upgrade_building_features = (
            upgrade_building_features
            # Air leakage reduction if high levels of infiltration
            .withColumn(
                "infiltration_ach50",
                F.when(F.col("infiltration_ach50") > 15, F.col("infiltration_ach50") * 0.6)
                .when(F.col("infiltration_ach50") > 10, F.col("infiltration_ach50") * 0.7)
                .when(F.col("infiltration_ach50") > 7.5, F.col("infiltration_ach50") * 0.8)
                .otherwise(F.col("infiltration_ach50")),
            )
        )

    if upgrade_id == 3:  # heat pump: min efficiency, electric backup
        # apply transforms for ducted vs ductless depending whether the home has ducts
        upgrade_building_features = upgrade_building_features.transform(
            update_hp_ducted_ductless, "ASHP, SEER 15, 9 HSPF", "ASHP, SEER 15, 9 HSPF"
        )
        # apply general heat pump transforms common to all heat pumps
        upgrade_building_features = upgrade_building_features.transform(upgrade_to_hp_general)
    # TODO: The ducted option is still a minisplit, do we neeed a way to represent this in the features?
    if upgrade_id in [4, 9]:  # heat pump: high efficiency, electric backup
        # apply transforms for ducted vs ductless depending whether the home has ducts
        upgrade_building_features = upgrade_building_features.transform(
            update_hp_ducted_ductless, "MSHP, SEER 24, 13 HSPF", "MSHP, SEER 29.3, 14 HSPF"
        )
        # apply general heat pump transforms common to all heat pumps
        upgrade_building_features = upgrade_building_features.transform(upgrade_to_hp_general)

    if upgrade_id == 15.04:  # Daikin ductless cchp
        # apply ductless heat pump transforms for multi vs single zone depending on the size of the home
        upgrade_building_features = get_hp_efficiencies_single_multi_zone_ductless(
            upgrade_building_features,
            single_zone_efficiency="Daikin MSHP SZ, SEER 22.05, 10.64 HSPF",
            multi_zone_efficiency="Daikin MSHP MZ, SEER 22.05, 11.2 HSPF",
        )

    if upgrade_id == 15.05:  # Carrier performance ductless cchp
        # apply ductless heat pump transforms for multi vs single zone depending on the size of the home
        upgrade_building_features = get_hp_efficiencies_single_multi_zone_ductless(
            upgrade_building_features,
            single_zone_efficiency="Carrier MSHP SZ, SEER 24.255, 13.104 HSPF",
            multi_zone_efficiency="Carrier MSHP MZ, SEER 26.25, 10.64 HSPF",
        )

    if upgrade_id == 15.06:  # York ductless cchp
        # apply ductless heat pump transforms for multi vs single zone depending on the size of the home
        upgrade_building_features = get_hp_efficiencies_single_multi_zone_ductless(
            upgrade_building_features,
            single_zone_efficiency="York MSHP SZ, SEER 23.625, 11.21 HSPF",
            multi_zone_efficiency="York MSHP MZ, SEER 25.36, 11.2 HSPF",
        )

    if upgrade_id == 15.08:  # Daikin ducted cc MSHP
        # apply ducted heat pump to only homes with ducts
        # Split into two DataFrames:
        df_with_ducts = upgrade_building_features.filter(F.col("has_ducts"))
        df_without_ducts = upgrade_building_features.filter(~F.col("has_ducts"))

        # Apply transformations only to df_with_ducts
        df_with_ducts_transformed = (
            df_with_ducts.withColumn(
                "heating_efficiency_nominal_percentage",
                extract_heating_efficiencyUDF(F.lit("Daikin 7 series ASHP, SEER 20, 10.35 HSPF")),
            )
            .withColumn(
                "cooling_efficiency_eer",
                extract_cooling_efficiencyUDF(F.lit("Daikin 7 series ASHP, SEER 20, 10.35 HSPF")),
            )
            .withColumn("has_ducted_heating", F.lit(True))
            .withColumn("has_ducted_cooling", F.lit(True))
            # Add column to join to performance curve metrics on
            .withColumn("hvac_heating_efficiency", F.lit("Daikin 7 series ASHP, SEER 20, 10.35 HSPF"))
            .transform(upgrade_to_hp_general, "HERS")
            .transform(remove_setbacks)
        )

        # Union transformed and unchanged DataFrames
        upgrade_building_features = df_with_ducts_transformed.union(df_without_ducts.transform(remove_setbacks))

    # all ductless cold climate heat pumps for rfp
    if upgrade_id in [15.04, 15.05, 15.06]:
        # apply general heat pump transforms common to all heat pumps
        upgrade_building_features = upgrade_building_features.transform(upgrade_to_hp_general, "HERS")
        # remove setbacks
        upgrade_building_features = remove_setbacks(upgrade_building_features)

    if upgrade_id in [11.05, 11.07, 13.01, 13.02]:
        # apply transforms for ducted vs ductless depending whether the home has ducts
        upgrade_building_features = upgrade_building_features.transform(
            update_hp_ducted_ductless, "ASHP, SEER 18, 10 HSPF", "ASHP, SEER 18, 10.5 HSPF"
        )
        # apply general heat pump transforms common to all heat pumps
        upgrade_building_features = upgrade_building_features.transform(upgrade_to_hp_general, "HERS")
        # remove setbacks
        upgrade_building_features = remove_setbacks(upgrade_building_features)

    if upgrade_id in [6, 9]:
        upgrade_building_features = (
            upgrade_building_features.withColumn("water_heater_fuel", F.lit("Electricity"))
            .withColumn(
                "water_heater_efficiency",
                F.when(  # electric tankless don't get upgraded due to likely size constraints
                    F.col("water_heater_efficiency") == "Electric Tankless",
                    F.col("water_heater_efficiency"),
                )
                .when(F.col("n_bedrooms") <= 3, F.lit("Electric Heat Pump, 50 gal, 3.45 UEF"))
                .when(F.col("n_bedrooms") == 4, F.lit("Electric Heat Pump, 66 gal, 3.35 UEF"))
                .otherwise(F.lit("Electric Heat Pump, 80 gal, 3.45 UEF")),
            )
            .transform(add_water_heater_features)
        )

    if upgrade_id == 9:
        # update to heat pump dryer and induction range
        upgrade_building_features = (
            upgrade_building_features.withColumn("clothes_dryer_fuel", F.lit("Electricity"))
            .withColumn("has_heat_pump_dryer", F.lit(True))
            .withColumn("cooking_range_fuel", F.lit("Electricity"))
            .withColumn("has_induction_range", F.lit(True))
        )

    # Add in heat pump performance curve params using pre-defined specs, or impute for building samples that do not have a hp
    upgrade_building_features = upgrade_building_features.join(
        HEAT_PUMP_PERFORMANCE_CURVE_DF, on="hvac_heating_efficiency", how="left"
    )
    # For samples without a heat pump, impute all COP columns with heating_efficiency_nominal_percentage
    # and all capacity retention columns with 1
    upgrade_building_features = fill_null_with_column(
        upgrade_building_features,
        source_column="heating_efficiency_nominal_percentage",
        columns_to_fill=["min_cop_47f", "min_cop_17f", "min_cop_5f", "max_cop_47f", "max_cop_17f", "max_cop_5f"],
    )
    upgrade_building_features = upgrade_building_features.fillna(
        1,
        subset=[
            "min_capacity_retention_47f",
            "min_capacity_retention_17f",
            "min_capacity_retention_5f",
            "max_capacity_retention_47f",
            "max_capacity_retention_17f",
            "max_capacity_retention_5f",
        ],
    )

    # add indicator features for presence of fuels (not including electricity)
    upgrade_building_features = (
        upgrade_building_features.withColumn("appliance_fuel_arr", F.array(APPLIANCE_FUEL_COLS))
        .withColumn("gas_misc_appliance_indicator_arr", F.array(GAS_APPLIANCE_INDICATOR_COLS))
        .withColumn(
            "has_natural_gas_appliance",
            (
                F.array_contains("appliance_fuel_arr", "Natural Gas")
                | F.array_contains("gas_misc_appliance_indicator_arr", True)
            ),
        )
        .withColumn("has_fuel_oil_appliance", F.array_contains("appliance_fuel_arr", "Fuel Oil"))
        .withColumn("has_propane_appliance", F.array_contains("appliance_fuel_arr", "Propane"))
        .drop(  # drop columns that were only used for upgrade lookups
            "insulation_wall",
            "ducts",
            "water_heater_efficiency",
            "appliance_fuel_arr",
            "gas_misc_appliance_indicator_arr",
        )
    )

    return upgrade_building_features


def build_upgrade_metadata_table(baseline_building_features: DataFrame) -> DataFrame:
    """
    Applied upgrade logic to baseline features table to create a DataFrame with features for each supported upgrade.

    This function iterates over each upgrade specified in `SUPPORTED_UPGRADES`,
    applies these upgrades to the baseline building metadata, and then unions the resulting DataFrames
    to create a comprehensive DataFrame that includes the baseline and all upgrades.

    Args:
        building_features_baseline (DataFrame): A Spark DataFrame containing baseline building metadata
          for a set of building samples, with the primary key (building_id, building_set)
    Returns:
        DataFrame: A Spark DataFrame containing building metadata for each upgrade including baseline.
    """
    # Get names, upgrade ids and name of baseline building set for each upgrade
    upgrade_rows = (
        upgrades_df(spark)
        .where(F.col("name").isin(SUPPORTED_UPGRADES))
        .select("name", "upgrade_id", "building_set")
        .collect()
    )

    # Iterate through each and apply the upgrade logic and add the name of the upgrade
    upgraded_dfs = []
    for row in upgrade_rows:
        upgrade_name, upgrade_id, building_set = row["name"], float(row["upgrade_id"]), row["building_set"]
        df = apply_upgrades(
            baseline_building_features=baseline_building_features.where(F.col("building_set") == building_set),
            upgrade_id=upgrade_id,
        ).withColumn("upgrade_name", F.lit(upgrade_name))
        upgraded_dfs.append(df)
    # Union into one table
    return reduce(DataFrame.unionByName, upgraded_dfs)


def drop_non_upgraded_samples(building_features: DataFrame, check_applicability_logic_against_version=None):
    """
    Drop upgrade records that had no changed features and therefore weren't upgraded.

    Note that a record is are marked as a non-upgraded duplicate if the sample's metadata is
    identical that of a lower upgrade. For example, if the metadata is identical for say 11.05
    and 13.01 for a given sample, the 13.01 record will be dropped.

    Args:
        building_metadata_upgrades (DataFrame): The DataFrame containing building metadata upgrades.
        check_applicability_logic_against_version (str, optional): If passed, check whether the applicabilitity logic
                matches between the metadata (i.e, non-unique set of metadata) the applicability flag output by the
                simulation for the given version number. Should only be passed if running on Resstock EUSS data. Defaults to None.

    Returns:
        DataFrame: The DataFrame with non-upgraded samples dropped.

    Raises:
        ValueError: If check_applicability_logic=True and the applicability logic
        does not match between the features and targets. Upgrade 13.01 is ignored.

    """
    partition_cols = building_features.drop("upgrade_id", "upgrade_name").columns
    w = Window.partitionBy(partition_cols).orderBy(F.asc("upgrade_id"))

    # partition by features-- if more them one row is in the partition,
    # then it is a duplicate, meaning an upgrade was not applied, so mark it as such
    # so mark any upgrade rows (upgrade > 0) as duplicate
    building_features_applicability_flag = (
        building_features.withColumn("rank", F.rank().over(w))
        .withColumn("applicability", F.col("rank") == 1)
        .drop("rank")
    )

    if check_applicability_logic_against_version is not None:
        # test that the applicability logic matches between the features and targets
        # we ignore a few RASstock upgrades since they are all flagged as True in the output table
        # even though many do not have the insulation upgrade applied and are therefore identical to previous upgrades
        applicability_compare = building_features_applicability_flag.alias("features").join(
            spark.table(f"{ANNUAL_OUTPUTS_TABLE}_{check_applicability_logic_against_version}")
            .select("upgrade_id", "building_id", "applicability")
            .alias("targets"),
            on=["upgrade_id", "building_id"],
        )
        mismatch_count = (
            applicability_compare.where(F.col("features.applicability") != F.col("targets.applicability"))
            .where(~F.col("upgrade_id").isin([13.01, 13.02, 11.05, 11.07]))
            .count()
        )
        if mismatch_count > 0:
            (
                applicability_compare.where(F.col("features.applicability") != F.col("targets.applicability"))
                .withColumnRenamed("`features.applicability`", "features_applicability")
                .withColumnRenamed("`targets.applicability`", "targets_applicability")
            ).display()
            raise ValueError(
                f"{mismatch_count} cases where applicability based on metadata and simulation applicability flag\
                      do not match"
            )

    # drop feature rows where upgrade was not applied
    return building_features_applicability_flag.where(F.col("applicability")).drop("applicability")


#  -- functions to construct weather city file index -- #
def create_string_indexer(df: DataFrame, column_name: str) -> StringIndexer:
    """
    Create and fit a StringIndexer for the distinct values in a given column.

    Args:
        df (spark.DataFrame): The DataFrame containing the column.
        column_name (str): The name of the column for which to create the index mapping.

    Returns:
        The fitted StringIndexer.
    """
    # Create a StringIndexer for the given column
    indexer = StringIndexer(
        inputCol=column_name, outputCol=f"{column_name}_index", stringOrderType="alphabetAsc", handleInvalid="skip"
    )

    # Fit the indexer to the DataFrame
    fitted_indexer = indexer.fit(df)

    return fitted_indexer


def fit_weather_city_index(df_to_fit: DataFrame):
    # Create the StringIndexer
    return create_string_indexer(df_to_fit.drop("weather_file_city_index"), "weather_file_city")


def transform_weather_city_index(weather_file_city_indexer: StringIndexer, df_to_transform: DataFrame):
    return weather_file_city_indexer.transform(df_to_transform).withColumn(
        "weather_file_city_index", F.col("weather_file_city_index").cast("int")
    )
