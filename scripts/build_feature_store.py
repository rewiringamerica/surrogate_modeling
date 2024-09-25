# Databricks notebook source
# MAGIC %md # Build Surrogate Model Feature Stores
# MAGIC
# MAGIC ### Goal
# MAGIC Transform surrogate model features (building metadata and weather) and write to feature store.
# MAGIC
# MAGIC ### Process
# MAGIC * Transform building metadata into features and subset to features of interest
# MAGIC * Apply upgrade logic to building metadata features
# MAGIC * Pivot weather data into wide vector format with pkey `weather_file_city` and a 8760-length timeseries vector for each weather feature column
# MAGIC * Write building metadata features and weather features to feature store tables
# MAGIC
# MAGIC ### I/Os
# MAGIC
# MAGIC ##### Inputs:
# MAGIC - `ml.surrogate_model.building_simulation_outputs_annual`: Building simulation outputs indexed by (building_id, upgrade_id)
# MAGIC - `ml.surrogate_model.building_metadata`: Building metadata indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_data_hourly`: Hourly weather data indexed by (weather_file_city, hour datetime)
# MAGIC
# MAGIC ##### Outputs:
# MAGIC - `ml.surrogate_model.building_features`: Building metadata features indexed by (building_id)
# MAGIC - `ml.surrogate_model.weather_features_hourly`: Weather features indexed by (weather_file_city) with a 8760-length timeseries vector for each weather feature column
# MAGIC
# MAGIC ### TODOs:
# MAGIC
# MAGIC #### Outstanding
# MAGIC
# MAGIC #### Future Work
# MAGIC - Updates to the feature table should merge and not overwrite, and in general transformation that are a hyperparameter of the model (i.e, that we may want to vary in different models) should be done downstream of this table. Sorting out exactly which transformations should happen in each of the `build_dataset`, `build_feature_store` and `model_training` files is still a WIP.
# MAGIC
# MAGIC ---
# MAGIC Cluster/ User Requirements
# MAGIC - Access Mode: Single User or Shared (Not No Isolation Shared)
# MAGIC - Runtime: >= Databricks Runtime 13.2 for ML or above (or >= Databricks Runtime 13.2 +  `%pip install databricks-feature-engineering`)
# MAGIC - `USE CATALOG`, `CREATE SCHEMA` privleges on the `ml` Unity Catalog (Ask Miki for access if permission is denied)

# COMMAND ----------

# DBTITLE 1,Imports
import re
from functools import reduce
from itertools import chain
from typing import Dict

from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F
from databricks.feature_engineering import FeatureEngineeringClient
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

# COMMAND ----------

# MAGIC %md ## Feature Transformation

# COMMAND ----------

# MAGIC %md ### Building Features

# COMMAND ----------

# MAGIC %md #### Baseline
# MAGIC
# MAGIC Refer to [Notion Page](https://www.notion.so/rewiringamerica/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=4#086a1d050b8c4094ad10e2275324668b) and [options.tsv](https://github.com/NREL/resstock/blob/run/euss/resources/options_lookup.tsv).
# MAGIC

# COMMAND ----------

# DBTITLE 1,Constants
# Constants
BTU_PER_WH = 3.413

EER_CONVERSION = {
    "EER": 1.0,
    "SEER": 0.875,
    "SEER2": 0.91,  # ~=SEER*1.04 (https://www.marathonhvac.com/seer-to-seer2/)
    "EER2": 1.04,
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

# mapping of window description to ufactor and shgc (solar heat gain coefficient) pulled from options.ts
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

# COMMAND ----------

# DBTITLE 1,Helper functions

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
def extract_r_value(construction_type: str, set_none_to_inf: bool = False) -> int:
    """Extract R-value from an unformatted string

    Assumption: all baseline walls have similar R-value of ~4.
    The returned value is for additional insulation only. Examples:
        Uninsulated brick, 3w, 12": ~4 (https://ncma.org/resource/rvalues-of-multi-wythe-concrete-masonry-walls/)
        Uninsulated wood studs: ~4 (assuming 2x4 studs and 1.25/inch (air gap has higher R-value than wood), 3.5*1.25=4.375)
        Hollow Concrete Masonry Unit, Uninsulated: ~4 per 6" (https://ncma.org/resource/rvalues-ufactors-of-single-wythe-concrete-masonry-walls/)

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
        raise ValueError(
            f"Cannot determine R-value of the construction type: "
            f"{construction_type}"
        )
    return int(m.group(1))


# TODO: figure out why raise value error is triggering
@udf(returnType=DoubleType())
def extract_cooling_efficiency(cooling_efficiency: str) -> float:
    """Convert a ResStock cooling efficiency into EER value

    >>> extract_cooling_efficiency('AC, SEER 13') / EER_CONVERSION['SEER']
    13.0
    >>> extract_cooling_efficiency('ASHP, SEER 20, 7.7 HSPF') / EER_CONVERSION['SEER']
    20.0
    >>> extract_cooling_efficiency('Room AC, EER 10.7')
    10.7
    >>> extract_cooling_efficiency('Shared Cooling')
    13.0
    >>> extract_cooling_efficiency('None') >= 99
    True
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
            raise ValueError(
                f"Cannot extract cooling efficiency from: {cooling_efficiency}"
            )
    # else:
    #     return 0.
    # raise ValueError(
    #         f'Cannot extract cooling efficiency from: {cooling_efficiency}')


@udf(returnType=DoubleType())
def extract_heating_efficiency(heating_efficiency: str) -> int:
    """
    Extract heating efficiency from string and convert to nominal percent efficiency
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
        # raise ValueError(
        #     f"Cannot extract heating efficiency from: {heating_efficiency}"
        # )

    if efficiency.endswith("AFUE"):
        return number / 100
    if efficiency.endswith("HSPF"):
        return round(number / BTU_PER_WH, 3)

    # 'Other' - e.g. wood stove - is not supported
    return number / 100


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
    if ef_string == "None":
        return 99.0
    return float(ef_string.split(",")[0][3:])


@udf(IntegerType())
def get_water_heater_capacity_ashrae(
    n_bedrooms: int, n_bathrooms: float, is_electric: bool
) -> int:
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
        StructField(  # "Storage", "Heat Pump", or "Instantaneous"
            "water_heater_type", StringType(), True
        ),
        StructField(  # Capacity of the tank in gallons
            "water_heater_tank_volume_gal", IntegerType(), True
        ),
        StructField(  # Efficiency Factor (EF)
            "water_heater_efficiency_ef", DoubleType(), True
        ),
        StructField(  # Recovery Efficiency Factor (EF)
            "water_heater_recovery_efficiency_ef", DoubleType(), True
        ),
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
        df.withColumn(
            "wh_struct", get_water_heater_specs(F.col("water_heater_efficiency"))
        )
        .withColumn("water_heater_type", F.col("wh_struct.water_heater_type"))
        .withColumn(
            "water_heater_tank_volume_gal",
            F.col("wh_struct.water_heater_tank_volume_gal"),
        )
        .withColumn(
            "water_heater_efficiency_ef", F.col("wh_struct.water_heater_efficiency_ef")
        )
        .withColumn(
            "water_heater_recovery_efficiency_ef",
            F.col("wh_struct.water_heater_recovery_efficiency_ef"),
        )
        .drop("wh_struct")
    )


# COMMAND ----------

# DBTITLE 1,Mapping Expressions

# Make various mapping expressions
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

# COMMAND ----------

# DBTITLE 1,Building metadata feature transformation function

def transform_building_features() -> DataFrame:
    """
    Read and transform subset of building_metadata features for single family homes.
    """
    building_metadata_transformed = (
        spark.read.table("ml.surrogate_model.building_metadata")
        # add upgrade id for baseline
        .withColumn("upgrade_id", F.lit(0.0))
        # -- filter to occupied sf homes with modeled fuels and without shared HVAC systems -- #
        # sf homes only
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
        # other fuels are not modeled in resstock,
        # and this filter is sufficienct to remove units that have other fuels for any applaince
        .where(F.col("heating_fuel") != "Other Fuel")
        .where(F.col("water_heater_fuel") != "Other Fuel")
        # filter out vacant homes
        .where(F.col("vacancy_status") == "Occupied")
        # filter out homes with shared HVAC or water heating systems
        .where(
            (F.col("hvac_has_shared_system") == "None")
            & (F.col("water_heater_in_unit") == "Yes")
        )
        # -- structure transformations -- #
        .withColumn("n_bedrooms", F.col("bedrooms").cast("int"))
        .withColumn("n_bathrooms", F.col("n_bedrooms") / 2 + 0.5)  # based on docs
        .withColumn(
            "orientation_degrees", orientation_degrees_mapping[F.col("orientation")]
        )
        .withColumn(
            "garage_size_n_car",
            F.coalesce(F.split(F.col("geometry_garage"), " ")[0].cast("int"), F.lit(0)),
        )
        .withColumn(
            "geometry_wall_exterior_finish",
            F.regexp_replace("geometry_wall_exterior_finish", "Shingle, ", "Shingle-"),
        )
        .withColumn(
            "exterior_wall_material", F.split("geometry_wall_exterior_finish", ", ")[0]
        )
        .withColumn(
            "exterior_wall_color",
            F.coalesce(
                F.split("geometry_wall_exterior_finish", ", ")[1], F.lit("None")
            ),
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
            F.when(
                F.col("heating_appliance_type").contains("Wall Furnace"), "Wall Furnace"
            )
            .when(F.col("heating_appliance_type").contains("Furnace"), "Furnace")
            .when(F.col("heating_appliance_type").contains("Boiler"), "Boiler")
            .when(F.col("heating_appliance_type") == "", "None")
            .otherwise(F.trim(F.col("heating_appliance_type"))),
        )
        .withColumn(
            "heating_efficiency_nominal_percentage",
            extract_heating_efficiency(F.col("hvac_heating_efficiency")),
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
            extract_cooling_efficiency(F.col("cooling_efficiency_eer_str")),
        )
        .withColumn(
            "cooling_setpoint_degrees_f",
            extract_temp(F.col("cooling_setpoint")),
        )
        .withColumn(
            "cooling_setpoint_offset_magnitude_degrees_f",
            extract_temp(F.col("cooling_setpoint_offset_magnitude")),
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
        .withColumn(
            "has_water_heater_in_unit", yes_no_mapping[F.col("water_heater_in_unit")]
        )
        # -- duct/infiltration tranformations -- #
        .withColumn("has_ducts", yes_no_mapping[F.col("hvac_has_ducts")])
        .withColumn(
            "duct_insulation_r_value", extract_r_value(F.col("ducts"), F.lit(True))
        )
        .withColumn("duct_leakage_percentage", extract_percentage(F.col("ducts")))
        .withColumn(
            "infiltration_ach50", F.split(F.col("infiltration"), " ")[0].cast("int")
        )
        # -- insulation tranformations -- #
        .withColumn("wall_material", F.split(F.col("insulation_wall"), ",")[0])
        .withColumn(
            "insulation_wall_r_value", extract_r_value(F.col("insulation_wall"))
        )
        .withColumn(
            "insulation_foundation_wall_r_value",
            extract_r_value(F.col("insulation_foundation_wall")),
        )
        .withColumn(
            "insulation_slab_r_value", extract_r_value(F.col("insulation_slab"))
        )
        .withColumn(
            "insulation_rim_joist_r_value",
            extract_r_value(F.col("insulation_rim_joist")),
        )
        .withColumn(
            "insulation_floor_r_value", extract_r_value(F.col("insulation_floor"))
        )
        .withColumn(
            "insulation_ceiling_r_value", extract_r_value(F.col("insulation_ceiling"))
        )
        .withColumn(
            "insulation_roof_r_value", extract_r_value(F.col("insulation_roof"))
        )
        #  -- building type transformations -- #
        .withColumn(
            "is_attached",
            ~F.col("geometry_building_type_acs").isin(
                ["Single-Family Detached", "Mobile Home"]
            ),
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
                F.lit(1),
            ),
        )
        .withColumn(
            "is_middle_unit",
            (F.col("geometry_building_horizontal_location_sfa") == "Middle")
            | (F.col("geometry_building_horizontal_location_mf") == "Middle"),
        )
        # -- other appliances -- #
        .withColumn("has_ceiling_fan", F.col("ceiling_fan") != "None")
        .withColumn("clothes_dryer_fuel", F.split(F.col("clothes_dryer"), ",")[0])
        .withColumn(
            "clothes_washer_efficiency", F.split(F.col("clothes_washer"), ",")[0]
        )
        .withColumn("cooking_range_fuel", F.split(F.col("cooking_range"), ",")[0])
        .withColumn(
            "dishwasher_efficiency_kwh",
            F.coalesce(F.split(F.col("dishwasher"), " ")[0].cast("int"), F.lit(9999)),
        )
        .withColumn(
            "lighting_efficiency", luminous_efficiency_mapping[F.col("lighting")]
        )
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
        .withColumn(
            "refrigerator_efficiency_ef", extract_energy_factor(F.col("refrigerator"))
        )
        .withColumn("plug_load_percentage", extract_percentage(F.col("plug_loads")))
        .withColumn(
            "usage_level_appliances", low_medium_high_mapping[F.col("usage_level")]
        )
        # -- misc transformations -- #
        .withColumn(
            "climate_zone_temp",
            F.substring("ashrae_iecc_climate_zone_2004", 1, 1).astype("int"),
        )
        .withColumn(
            "climate_zone_moisture", F.substring("ashrae_iecc_climate_zone_2004", 2, 1)
        )
        .withColumn(
            "neighbor_distance_ft",
            F.when(F.col("neighbors") == "Left/Right at 15ft", 15.0)
            .when(F.col("neighbors") == "None", 9999.0)
            .otherwise(F.col("neighbors").cast("double")),
        )
        .withColumn(
            "n_occupants",
            F.when(F.col("occupants") == "10+", 11).otherwise(
                F.col("occupants").cast("int")
            ),
        )
        .withColumn("vintage", extract_vintage(F.col("vintage")))
        # align names for methane gas across applainces
        .replace(
            {
                "Natural Gas": "Methane Gas",
                "Gas": "Methane Gas",
                "Electric": "Electricity",
            },
            subset=APPLIANCE_FUEL_COLS,
        )
        # subset to all possible features of interest
        .select(
            # primary keys
            "building_id",
            F.col("upgrade_id").cast("double"),
            # foreign key
            "weather_file_city",
            # structure
            "n_bedrooms",
            "n_bathrooms",
            F.col("geometry_attic_type").alias("attic_type"),
            "sqft",
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
            "has_ducted_heating",
            "heating_efficiency_nominal_percentage",
            "heating_setpoint_degrees_f",
            "heating_setpoint_offset_magnitude_degrees_f",
            # cooling
            "ac_type",
            "cooled_space_percentage",
            "cooling_efficiency_eer",
            "cooling_setpoint_degrees_f",
            "cooling_setpoint_offset_magnitude_degrees_f",
            # water heater
            "water_heater_efficiency",  # only used for applying upgrades, gets dropped later
            "water_heater_fuel",
            "water_heater_type",
            "water_heater_tank_volume_gal",
            "water_heater_efficiency_ef",
            "water_heater_recovery_efficiency_ef",
            "has_water_heater_in_unit",
            # ducts
            "ducts",  # only used for applying upgrades, gets dropped later
            "has_ducts",
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
            "clothes_washer_efficiency",
            "cooking_range_fuel",
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
    return building_metadata_transformed


# COMMAND ----------

# DBTITLE 1,Transform building metadata
building_metadata_transformed = transform_building_features()

# COMMAND ----------

# MAGIC %md #### Upgrades
# MAGIC
# MAGIC Refer to [Notion Page](https://www.notion.so/rewiringamerica/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=4#3141dfeeb07144da9fe983b2db13b6d3), [ResStock docs](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf), and [upgrade.yml](https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_10k.yml).

# COMMAND ----------

# DBTITLE 1,Apply upgrade functions
# Mapping of climate zone temperature  -> threshold, insulation
# where climate zone temperature is the first character in the ASHRAE IECC climate zone
# ('1', 13, 30) means units in climate zones 1A (1-anything) with R13 insulation or less are upgraded to R30
BASIC_ENCLOSURE_INSULATION = spark.createDataFrame(
    [
        ("1", 13, 30),
        ("2", 30, 49),
        ("3", 30, 49),
        ("4", 38, 60),
        ("5", 38, 60),
        ("6", 38, 60),
        ("7", 38, 60),
    ],
    ("climate_zone_temp", "existing_insulation_max_threshold", "insulation_upgrade"),
)


def upgrade_to_hp(
    baseline_building_features: DataFrame,
    ducted_efficiency: str,
    non_ducted_efficiency: str,
) -> DataFrame:
    """
    Upgrade the baseline building features to an air source heat pump (ASHP) with specified efficiencies.
    Note that all baseline hps in Resstock are lower efficiency than specified upgrade thresholds (<=SEER 15; <=HSPF 8.5)

    Args:
        baseline_building_features (DataFrame): The baseline building features.
        ducted_efficiency (str): The efficiency of the ducted heat pump.
        non_ducted_efficiency (str): The efficiency of the ductless heat pump.

    Returns:
        DataFrame: The upgraded building features DataFrame with the heat pump.
    """
    return (
        baseline_building_features.withColumn("heating_appliance_type", F.lit("ASHP"))
        .withColumn("heating_fuel", F.lit("Electricity"))
        .withColumn(
            "heating_efficiency_nominal_percentage",
            F.when(
                F.col("has_ducts"),
                extract_heating_efficiency(F.lit(ducted_efficiency)),
            ).otherwise(extract_heating_efficiency(F.lit(non_ducted_efficiency))),
        )
        .withColumn("has_ducted_heating", F.col("has_ducts"))
        .withColumn(
            "cooling_efficiency_eer",
            F.when(
                F.col("has_ducts"),
                extract_cooling_efficiency(F.lit(ducted_efficiency)),
            ).otherwise(extract_cooling_efficiency(F.lit(non_ducted_efficiency))),
        )
        .withColumn("ac_type", F.lit("Heat Pump"))
        .withColumn("cooled_space_percentage", F.lit(1.0))
    )


def apply_upgrades(baseline_building_features: DataFrame, upgrade_id: int) -> DataFrame:
    """
    Modify building features to reflect the upgrade. Source:
    https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf
    In case of contradictions, consult: https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_10k.yml.

    Args:
          baseline_building_features: (DataFrame) building features coming from metadata.
          upgrade_id: (int)

    Returns:
          DataFrame: building_features, augmented to reflect the upgrade.
    """
    # Raise an error if an unsupported upgrade ID is provided
    if upgrade_id not in [0, 1, 3, 4, 6, 8.1, 8.2, 9]:
        raise ValueError(f"Upgrade id={upgrade_id} is not yet supported")

    upgrade_building_features = (
        baseline_building_features.withColumn("upgrade_id", F.lit(upgrade_id))
        .withColumn("has_heat_pump_dryer", F.lit(False))
        .withColumn("has_induction_range", F.lit(False))
    )

    if upgrade_id == 0:  # baseline: return as is
        pass

    if upgrade_id in [1, 9]:  # basic enclosure
        upgrade_building_features = (
            upgrade_building_features
            # Upgrade insulation of ceiling
            # Map the climate zone number to the insulation params for the upgrade
            .join(BASIC_ENCLOSURE_INSULATION, on="climate_zone_temp")
            # Attic floor insulation if current insulation is below threshold
            .withColumn(
                "insulation_ceiling_r_value",
                F.when(
                    (F.col("attic_type") == "Vented Attic")
                    & (
                        F.col("insulation_ceiling_r_value")
                        <= F.col("existing_insulation_max_threshold")
                    ),
                    F.col("insulation_upgrade"),
                ).otherwise(F.col("insulation_ceiling_r_value")),
            )
            .drop("insulation_upgrade", "existing_insulation_max_threshold")
            # Air leakage reduction if high levels of infiltration
            .withColumn(
                "infiltration_ach50",
                F.when(
                    F.col("infiltration_ach50") >= 15, F.col("infiltration_ach50") * 0.7
                ).otherwise(F.col("infiltration_ach50")),
            )
            # Duct sealing: update duct leakage rate to at most 10% and insulation to at least R-8,
            # with the exeption of "0% or 30% Leakage, Uninsulated" which does not get upgrade applied
            .withColumn(
                "duct_leakage_percentage",
                F.when(
                    (
                        (F.col("has_ducts"))
                        & ~(
                            F.col("ducts").isin(
                                ["0% Leakage, Uninsulated", "30% Leakage, Uninsulated"]
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
                            F.col("ducts").isin(
                                ["0% Leakage, Uninsulated", "30% Leakage, Uninsulated"]
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
                    extract_r_value(F.lit("Wood Stud, R-13")),
                ).otherwise(F.col("insulation_wall_r_value")),
            )
        )

    if upgrade_id == 3:  # heat pump: min efficiency, electric backup
        upgrade_building_features = upgrade_building_features.transform(
            upgrade_to_hp, "Heat Pump, SEER 15, 9 HSPF", "Heat Pump, SEER 15, 9 HSPF"
        )

    if upgrade_id in [4, 9]:  # heat pump: high efficiency, electric backup
        upgrade_building_features = upgrade_building_features.transform(
            upgrade_to_hp,
            "Heat Pump, SEER 24, 13 HSPF",
            "Heat Pump, SEER 29.3, 14 HSPF",
        )
    if upgrade_id in [6, 9]:
        upgrade_building_features = upgrade_building_features.withColumn(
            "water_heater_efficiency",
            F.when(  # electric tankless don't get upgraded due to likely size constraints
                F.col("water_heater_efficiency") == "Electric Tankless",
                F.col("water_heater_efficiency"),
            )
            .when(
                F.col("n_bedrooms") <= 3, F.lit("Electric Heat Pump, 50 gal, 3.45 UEF")
            )
            .when(
                F.col("n_bedrooms") == 4, F.lit("Electric Heat Pump, 66 gal, 3.35 UEF")
            )
            .otherwise(F.lit("Electric Heat Pump, 80 gal, 3.45 UEF")),
        ).transform(add_water_heater_features)

    if upgrade_id in [8.1, 9]:
        upgrade_building_features = upgrade_building_features.withColumn(
            "clothes_dryer_fuel", F.lit("Electricity")
        ).withColumn("has_heat_pump_dryer", F.lit(True))
    if upgrade_id in [8.2, 9]:
        upgrade_building_features = upgrade_building_features.withColumn(
            "cooking_range_fuel", F.lit("Electricity")
        ).withColumn("has_induction_range", F.lit(True))

    # add indicator features for presence of fuels (not including electricity)
    upgrade_building_features = (
        upgrade_building_features.withColumn(
            "appliance_fuel_arr", F.array(APPLIANCE_FUEL_COLS)
        )
        .withColumn(
            "gas_misc_appliance_indicator_arr", F.array(GAS_APPLIANCE_INDICATOR_COLS)
        )
        .withColumn(
            "has_methane_gas_appliance",
            (
                F.array_contains("appliance_fuel_arr", "Methane Gas")
                | F.array_contains("gas_misc_appliance_indicator_arr", True)
            ),
        )
        .withColumn(
            "has_fuel_oil_appliance", F.array_contains("appliance_fuel_arr", "Fuel Oil")
        )
        .withColumn(
            "has_propane_appliance", F.array_contains("appliance_fuel_arr", "Propane")
        )
        .drop(  # drop columns that were only used for upgrade lookups
            "insulation_wall",
            "ducts",
            "water_heater_efficiency",
            "appliance_fuel_arr",
            "gas_misc_appliance_indicator_arr",
        )
    )

    return upgrade_building_features


# COMMAND ----------

# DBTITLE 1,Apply upgrade logic to baseline features
# create a metadata df for baseline and each HVAC upgrade
upgrade_ids = [0.0, 1.0, 3.0, 4.0, 6.0, 9.0]
building_metadata_upgrades = reduce(
    DataFrame.unionByName,
    [
        apply_upgrades(
            baseline_building_features=building_metadata_transformed, upgrade_id=upgrade
        )
        for upgrade in upgrade_ids
    ],
)

# COMMAND ----------

# DBTITLE 1,Drop rows where upgrade was not applied
# read in outputs so that we can test applicability logic
annual_outputs = spark.table("ml.surrogate_model.building_simulation_outputs_annual")

# drop upgrades that had no unchanged features and therefore weren't upgraded
partition_cols = building_metadata_upgrades.drop("upgrade_id").columns
w = Window.partitionBy(partition_cols).orderBy(F.asc("upgrade_id"))

# partition by features-- if more them one row is in the partition,
# then it is a duplicate, meaning an upgrade was not applied, so mark it as such
# so mark any upgrade rows (upgrade > 0) as duplicate
building_metadata_upgrades_applicability_flag = (
    building_metadata_upgrades.withColumn("rank", F.rank().over(w))
    .withColumn("applicability", F.col("rank") == 1)
    .drop("rank")
)

# test that the applicability logic matches between the features and targets
applicability_compare = building_metadata_upgrades_applicability_flag.alias(
    "features"
).join(
    annual_outputs.select("upgrade_id", "building_id", "applicability").alias(
        "targets"
    ),
    on=["upgrade_id", "building_id"],
)
assert (
    applicability_compare.where(
        F.col("features.applicability") != F.col("targets.applicability")
    ).count()
    == 0
)
# #display mismatching cases if assert fails
# applicability_compare.where(F.col('features.applicability') != F.col('targets.applicability')).display()

# drop feature rows where upgrade was not applied
building_metadata_applicable_upgrades = (
    building_metadata_upgrades_applicability_flag.where(F.col("applicability")).drop(
        "applicability"
    )
)

# COMMAND ----------

# MAGIC %md #### Summary

# COMMAND ----------

# DBTITLE 1,Check if there are any Null Features
# count how many null vals are in each column
null_counts = building_metadata_upgrades.select(
    [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in building_metadata_upgrades.columns
    ]
)
# Collect the results as a dictionary
null_counts_dict = null_counts.collect()[0].asDict()
# select any items that have non-zero count
null_count_items = {k: v for k, v in null_counts_dict.items() if v > 0}
assert len(null_count_items) == 0, f"Columns with null values: {null_count_items}"

# COMMAND ----------

# DBTITLE 1,Print out feature counts
pkey_cols = ["weather_file_city", "upgrade_id", "building_id"]
string_columns = [
    field.name
    for field in building_metadata_upgrades.schema.fields
    if isinstance(field.dataType, StringType) and field.name not in pkey_cols
]
non_string_columns = [
    field.name
    for field in building_metadata_upgrades.schema.fields
    if not isinstance(field.dataType, StringType) and field.name not in pkey_cols
]
# count how many distinct vals there are for each categorical feature
distinct_string_counts = building_metadata_upgrades.select(
    [F.countDistinct(c).alias(c) for c in string_columns]
)
# Collect the results as a dictionary
distinct_string_counts_dict = distinct_string_counts.collect()[0].asDict()
# print the total number of features:
print(f"Building Metadata Features: {len(non_string_columns + string_columns)}")
print(f"\tNumeric Features: {len(non_string_columns)}")
print(f"\tCategorical Features: {len(string_columns)}")
print(
    f"\tFeature Dimensionality: {len(non_string_columns) + sum(distinct_string_counts_dict.values()) }"
)

# COMMAND ----------

# MAGIC %md ### Weather Features

# COMMAND ----------

# DBTITLE 1,Weather feature transformation function

def transform_weather_features() -> DataFrame:
    """
    Read and transform weather timeseries table. Pivot from long format indexed by (weather_file_city, hour)
    to a table indexed by weather_file_city with a 8760 len array timeseries for each weather feature column

    Returns:
        DataFrame: wide(ish) format dataframe indexed by weather_file_city with timeseries array for each weather feature
    """
    weather_df = spark.read.table("ml.surrogate_model.weather_data_hourly")
    weather_pkeys = ["weather_file_city"]

    weather_data_arrays = weather_df.groupBy(weather_pkeys).agg(
        *[
            F.collect_list(c).alias(c)
            for c in weather_df.columns
            if c not in weather_pkeys + ["datetime_formatted"]
        ]
    )
    return weather_data_arrays


# COMMAND ----------

# DBTITLE 1,Transform weather features
weather_data_transformed = transform_weather_features()

# COMMAND ----------

# DBTITLE 1,Create and apply string indexer to generate weather file city index
# Create the StringIndexer
indexer = StringIndexer(
    inputCol="weather_file_city",
    outputCol="weather_file_city_index",
    stringOrderType="alphabetAsc",
)

weather_file_city_indexer = indexer.fit(weather_data_transformed)

weather_data_indexed = weather_file_city_indexer.transform(
    weather_data_transformed
).withColumn("weather_file_city_index", F.col("weather_file_city_index").cast("int"))

building_metadata_applicable_upgrades_with_weather_file_city_index = (
    weather_file_city_indexer.transform(
        building_metadata_applicable_upgrades
    ).withColumn(
        "weather_file_city_index", F.col("weather_file_city_index").cast("int")
    )
)

# COMMAND ----------

# MAGIC %md ## Create Feature Store
# MAGIC

# COMMAND ----------

# MAGIC %md ### Create/Use schema in catalog in the Unity Catalog MetaStore
# MAGIC
# MAGIC To use an existing catalog, you must have the `USE CATALOG` privilege on the catalog.
# MAGIC To create a new schema in the catalog, you must have the `CREATE SCHEMA` privilege on the catalog.

# COMMAND ----------

# DBTITLE 1,Check if you have access on ml catalog
# MAGIC %sql
# MAGIC -- if you do not see `ml` listed here, this means you do not have permissions
# MAGIC SHOW CATALOGS

# COMMAND ----------

# DBTITLE 1,Set up catalog and schema
# MAGIC %sql
# MAGIC -- Use existing catalog:
# MAGIC USE CATALOG ml;
# MAGIC -- Create a new schema
# MAGIC CREATE SCHEMA IF NOT EXISTS surrogate_model;
# MAGIC USE SCHEMA surrogate_model;

# COMMAND ----------

# MAGIC %md ### Create/modify the feature stores

# COMMAND ----------

# MAGIC %sql
# MAGIC -- the following code will upsert. To overwrite, uncommnet this line to first drop the table
# MAGIC -- DROP TABLE ml.surrogate_model.building_features

# COMMAND ----------

# DBTITLE 1,Create a FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Write out building metadata feature store
table_name = "ml.surrogate_model.building_features"
df = building_metadata_applicable_upgrades_with_weather_file_city_index
if spark.catalog.tableExists(table_name):
    fe.write_table(name=table_name, df=df, mode="merge")
else:
    fe.create_table(
        name=table_name,
        primary_keys=["building_id", "upgrade_id", "weather_file_city"],
        df=df,
        schema=df.schema,
        description="building metadata features",
    )

# COMMAND ----------

# DBTITLE 1,Write out weather data feature store
table_name = "ml.surrogate_model.weather_features_hourly"
df = weather_data_indexed
if spark.catalog.tableExists(table_name):
    fe.write_table(
        name=table_name,
        df=df,
        mode="merge",
    )
else:
    fe.create_table(
        name=table_name,
        primary_keys=["weather_file_city"],
        df=df,
        schema=df.schema,
        description="hourly weather timeseries array features",
    )
