"""Tests for building surrogate model features"""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def spark_session():
    """Provide a spark session to use in tests."""
    spark = SparkSession.builder.getOrCreate()
    yield spark


@pytest.mark.parametrize(
    "upgrade_id, input, expected",
    [
        #  (1, 
        #   {"attic_type" : "Vented Attic"},
        #   {"insulation_ceiling_r_value": 60})
        (0, 
          {},
          {"has_heat_pump_dryer" : False,
           "has_induction_range" : False})
    ]
)
def test_the_thing(upgrade_id, input, expected):
    # start with a default building
    default_df = pd.DataFrame({
        'climate_zone_temp': [1],
        'attic_type': ['Vented Attic'],
        'insulation_ceiling_r_value': [10],
        'existing_insulation_max_threshold': [15],
        'infiltration_ach50': [20],
        'has_ducts': [True],
        'duct_leakage_percentage': [0.15],
        'duct_insulation_r_value': [6],
        'insulation_wall_r_value': [4],
        'insulation_wall': ['Wood Stud, Uninsulated'],
        'n_bedrooms': [3],
        'water_heater_efficiency': ['Electric Tankless'],
        'ducts': ['0% Leakage, Uninsulated']
        }
    )

    # override fields from the inputs
    for k,v in input.items():
        default_df[k] = v

    result = apply_upgrades(df, upgrade_id)

    expected["upgrade_id"]=upgrade_id
    for k,v in expected.items():
        assert result[k] == v