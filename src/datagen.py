import itertools
import logging
import math
import os
import re
from typing import Dict

from utils import file_cache

import numpy as np
import pandas as pd
import tensorflow as tf

# Constants
SEER_TO_EER = .875
# http://www.energyguru.com/EnergyEfficiencyInformation.htm
BTU_PER_WH = 3.414
HOURS_IN_A_YEAR = 8760  # 24*365, assuming a non-leap year

# Path to ResStock dataset
# TODO: replace with environment variables
# to access gs:// paths without explicitly providing credentials, run
# `gcloud auth application-default login` (only required once)
RESSTOCK_PATH = 'gs://the-cube/data/raw/nrel/end_use_load_profiles/2022/resstock_tmy3_release_1/'

BUILDING_METADATA_PARQUET_PATH = RESSTOCK_PATH + 'metadata_and_annual_results/national/parquet/baseline_metadata_only.parquet'
HOURLY_OUTPUT_PATH = RESSTOCK_PATH + 'timeseries_individual_buildings/by_state/upgrade={upgrade_id}/state={state}/{building_id}-{upgrade_id}.parquet'
# pattern of weather files path within RESSTOCK_PATH
# examples:
# `resstock_tmy3_release_1`, `resstock_tmy3_release_1.1`:
#       `.../weather/state={state}/{geoid}_TMY3.csv`
# `resstock_amy2018_release_1`, `resstock_amy2018_release_1.1`:
#       `.../weather/state={state}/{geoid}_f018.csv`
# `comstock_amy2018_release_2`:
#       `.../weather/amy2018/{geoid}_2018.csv`
WEATHER_FILES_PATH = RESSTOCK_PATH + 'weather/state={state}/{geoid}_TMY3.csv'
STATE_2NUM_CODE_TO_2LETTER = {  # Note: keys are intentionally strings to simplify parsing county geoid
    '01': 'AL',
    '02': 'AK',
    '04': 'AZ',
    '05': 'AR',
    '06': 'CA',
    '08': 'CO',
    '09': 'CT',
    '10': 'DE',
    '11': 'DC',
    '12': 'FL',
    '13': 'GA',
    '16': 'ID',
    '17': 'IL',
    '18': 'IN',
    '19': 'IA',
    '20': 'KS',
    '21': 'KY',
    '22': 'LA',
    '23': 'ME',
    '24': 'MD',
    '25': 'MA',
    '26': 'MI',
    '27': 'MN',
    '28': 'MS',
    '29': 'MO',
    '30': 'MT',
    '31': 'NE',
    '32': 'NV',
    '33': 'NH',
    '34': 'NJ',
    '35': 'NM',
    '36': 'NY',
    '37': 'NC',
    '38': 'ND',
    '39': 'OH',
    '40': 'OK',
    '41': 'OR',
    '42': 'PA',
    '44': 'RI',
    '45': 'SC',
    '46': 'SD',
    '47': 'TN',
    '48': 'TX',
    '49': 'UT',
    '50': 'VT',
    '51': 'VA',
    '53': 'WA',
    '54': 'WV',
    '55': 'WI',
    '56': 'WY',
}

CACHE_PATH = '.cache'
if not os.path.isdir(CACHE_PATH):
    logging.warning(f"Cache path {CACHE_PATH} does not exist. Attempting to create..")
    os.mkdir(CACHE_PATH)
    logging.warning("Success")


def vintage2age2000(vintage: str) -> int:
    """ vintage of the building in the year of 2000
    >>> vintage2age2000('<1940')
    70
    >>> vintage2age2000('1960s')
    40
    """
    vintage = vintage.strip()
    if vintage.startswith('<'):  # '<1940' bin in resstock
        return 70
    return 2000 - int(vintage[:4])


def extract_r_value(construction_type: str) -> int:
    """ Extract R-value from an unformatted string

    Assumption: all baseline walls have similar R-value of ~4.
    The returned value is for additional insulation only. Examples:
        Uninsulated brick, 3w, 12": ~4 (https://ncma.org/resource/rvalues-of-multi-wythe-concrete-masonry-walls/)
        Uninsulated wood studs: ~4 (assuming 2x4 studs and 1.25/inch (air gap has higher R-value than wood), 3.5*1.25=4.375)
        Hollow Concrete Masonry Unit, Uninsulated: ~4 per 6" (https://ncma.org/resource/rvalues-ufactors-of-single-wythe-concrete-masonry-walls/)

    >>> extract_r_value('Finished, R-13')
    13
    >>> extract_r_value('Brick, 12-in, 3-wythe, R-15')
    15
    >>> extract_r_value('CMU, 6-in Hollow, Uninsulated')
    0
    >>> extract_r_value('2ft R10 Under, Horizontal')
    10
    >>> extract_r_value('R-5, Exterior')
    5
    >>> extract_r_value('Ceiling R-19')
    19
    """
    lower = construction_type.lower()
    if lower == 'none' or 'uninsulated' in lower:
        return 0
    m = re.search(r"\br-?(\d+)\b", construction_type, flags=re.I)
    if not m:
        raise ValueError(
            f'Cannot determine R-value of the construction type: '
            f'{construction_type}'
        )
    return int(m.group(1))


def extract_cooling_efficiency(cooling_efficiency: str) -> float:
    """ Convert a ResStock cooling efficiency into EER value
    >>> extract_cooling_efficiency('AC, SEER 13')
    11.375
    >>> extract_cooling_efficiency('Heat Pump')
    11.375
    >>> extract_cooling_efficiency('Room AC, EER 10.7')
    10.7
    >>> extract_cooling_efficiency('None')
    99
    """
    ac_type = cooling_efficiency.split(", ", 1)[0].strip()
    efficiency = cooling_efficiency.rsplit(", ", 1)[-1].strip()
    if efficiency.startswith('EER'):
        return float(efficiency.rsplit(' ')[-1])
    if efficiency.startswith('SEER'):
        return float(efficiency.rsplit(' ')[-1]) * SEER_TO_EER
    if ac_type == 'Heat Pump':
        # a default value as we don't have anything else.
        # Min SEER for heat pumps is 13 by law, 13*.875 ~= 11.4
        return 13*SEER_TO_EER
    if ac_type == 'None':
        # insanely high efficiency to mimic a nonexistent colling
        return 99
    raise ValueError(
        f'Cannot extract cooling efficiency from: {cooling_efficiency}'
    )


def extract_heating_efficiency(heating_efficiency: str) -> int:
    """
    "Other" IS used in single family homes, "Shared Heating" seemingly isn't
    >>> extract_heating_efficiency('Fuel Furnace, 80% AFUE')
    80
    >>> extract_heating_efficiency('ASHP, SEER 15, 8.5 HSPF')
    248
    >>> extract_heating_efficiency('None') >= 999
    True
    >>> extract_heating_efficiency('Electric Baseboard, 100% Efficiency')
    100
    >>> extract_heating_efficiency('Other')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Cannot extract heating efficiency from: ...
    """
    efficiency = heating_efficiency.rsplit(", ", 1)[-1]
    if efficiency == "None":
        return 999

    try:
        number = float(efficiency.strip().split(" ", 1)[0].strip("%"))
    except ValueError:
        raise ValueError(
            f'Cannot extract heating efficiency from: {heating_efficiency}'
        )

    if efficiency.endswith("AFUE"):
        return int(number)
    if efficiency.endswith("HSPF"):
        return int(number*100/BTU_PER_WH)

    # 'Other' - e.g. wood stove - is not supported
    return int(number)


@file_cache(CACHE_PATH)
def _get_building_metadata():
    """ Helper function to retrieve and clean building metadata

    >>> metadata_df = _get_building_metadata()
    >>> isinstance(metadata_df, pd.DataFrame)
    True
    >>> metadata_df.shape[0] > 300000  # at least 300k buildings
    True
    >>> metadata_df.shape[1] > 10  # at least 10 columns
    True
    >>> metadata = metadata_df.iloc[0]
    >>> all(
    ...     not isinstance(value, str)
    ...     for col, value in metadata.items()
    ...     if col not in ('county', 'ashrae_iecc_climate_zone')
    ... )
    True
    """
    pq = pd.read_parquet(
        BUILDING_METADATA_PARQUET_PATH,
        columns=[
            # features used directly or transformed
            'in.sqft', 'in.bedrooms', 'in.geometry_stories',
            'in.vintage', 'in.geometry_building_number_units_mf',
            'in.geometry_building_number_units_sfa',
            # features to be used to join with other datasets
            'in.county',  # weather files
            # features that will be replaced with "reasonable assumptions"
            'in.occupants',
            # it's either ceiling or roof; only ~15K (<3%) have none
            'in.insulation_ceiling', 'in.insulation_roof',
            'in.insulation_wall', 'in.infiltration',
            'in.hvac_cooling_efficiency', 'in.hvac_heating_efficiency',
            # to be filtered on
            'in.has_pv', 'in.geometry_building_type_acs',
            # ashrae_iecc_climate_zone_2004_2_a_split splits 2A states into
            # two groups, otherwise it's the same
            'in.ashrae_iecc_climate_zone_2004'
        ],
    ).rename(
        # to make this code interchangeable with the spark tables
        columns={
            'in.sqft': 'sqft',
            'in.bedrooms': 'bedrooms',
            'in.geometry_stories': 'stories',
            'in.occupants': 'occupants',
            'in.county': 'county',
            'in.ashrae_iecc_climate_zone_2004': 'ashrae_iecc_climate_zone',
        }
    )
    pq.index.rename('building_id', inplace=True)

    pq = pq[
        (pq['in.geometry_building_type_acs'] == 'Single-Family Detached')
        & (pq['occupants'] != '10+')
        # sanity check; it's 1 for all single family detached
        # & (pq[
        #     ['in.geometry_building_number_units_mf', 'in.geometry_building_number_units_sfa']
        # ].replace('None', 1).max(axis=1).fillna(1).astype(int) == 1)
        # another sanity check; ResStock single family detached have 3 max
        & (pq['stories'] <= '5')
        # for some reason there are 14K 8194sqf single family detached homes
        & (pq['sqft'] < 8000)
        # Not sure how to model these yet
        & ~pq['in.hvac_heating_efficiency'].isin(['Other', 'Shared Heating'])
        & (pq['in.hvac_cooling_efficiency'] != 'Shared Cooling')
        # we'll get to solar, eventually - just not yet
        & (pq['in.has_pv'] == 'No')
    ]
    pq = pq.assign(
        age2000=pq['in.vintage'].map(vintage2age2000),
        bedrooms=pq['bedrooms'].astype(int),
        stories=pq['stories'].astype(int),
        occupants=pq['occupants'].astype(int),
        infiltration_ach50=pq['in.infiltration'].str.split().str[0].astype(int),
        insulation_wall=pq['in.insulation_wall'].map(extract_r_value),
        insulation_ceiling_roof=pq[
            ['in.insulation_ceiling', 'in.insulation_roof']
        ].map(extract_r_value).max(axis=1),
        cooling_efficiency_eer=pq['in.hvac_cooling_efficiency'].map(extract_cooling_efficiency),
        heating_efficiency=pq['in.hvac_heating_efficiency'].map(extract_heating_efficiency),
    ).drop(
        columns=[
            'in.vintage', 'in.geometry_building_type_acs',
            'in.has_pv', 'in.geometry_building_number_units_mf',
            'in.geometry_building_number_units_sfa',
            'in.infiltration', 'in.insulation_wall',
            'in.insulation_ceiling', 'in.insulation_roof',
            'in.hvac_cooling_efficiency', 'in.hvac_heating_efficiency',
        ]
    )

    # extra safety check to eliminate duplicate buildings
    # (not that there are any)
    return pq.loc[pq.index.drop_duplicates()]


class BuildingMetadataBuilder:
    """ A class to cache building metadata in memory.
    """
    _building_metadata = None
    supported_building_features = (
        'sqft', 'bedrooms', 'stories', 'occupants', 'age2000', 'county',
        'infiltration_ach50', 'insulation_wall', 'insulation_ceiling_roof',
        'cooling_efficiency_eer', 'heating_efficiency',
    )

    def __init__(self):
        """
        >>> builder = BuildingMetadataBuilder()
        >>> isinstance(builder.all(), pd.DataFrame)
        True
        >>> isinstance(builder.building_ids, np.ndarray)
        True
        >>> isinstance(builder(100066), pd.Series)
        True
        """
        pq = _get_building_metadata()
        self._building_metadata = pq[list(self.supported_building_features)]

    def all(self) -> pd.DataFrame:
        return self._building_metadata

    @property
    def building_ids(self) -> np.ndarray:
        # make a copy because otherwise np.shuffle will mess up the index
        return np.array(self.all().index.values)

    def __call__(self, building_id) -> pd.Series:
        return self._building_metadata.loc[building_id]


def get_state_code_from_county_geoid(county_geoid):
    """ Extract two-letter state code from a county geoid in ResStock format

    >>> get_state_code_from_county_geoid('G0200130')
    'AK'
    >>> get_state_code_from_county_geoid('G4200110')
    'PA'
    >>> get_state_code_from_county_geoid('G0000000')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Invalid state code ...
    >>> get_state_code_from_county_geoid('D0200130')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Unexpected county geoid format ...
    """
    match = re.match(r"G(\d{2})0(\d{3})0", county_geoid)
    if not match:
        raise ValueError(
            f"Unexpected county geoid format: {county_geoid}. Something of a "
            f"form `G<2-num state code>0<3-num county code>0` expected"
        )
    state_2num_code = match.group(1)
    if state_2num_code not in STATE_2NUM_CODE_TO_2LETTER:
        raise ValueError(
            f"Invalid state code (`{state_2num_code}`) in county geoid: "
            f"`{county_geoid}`. Values between `00` and `50` expected"
        )
    return STATE_2NUM_CODE_TO_2LETTER[state_2num_code]


@file_cache(CACHE_PATH)
def get_hourly_outputs(building_id, upgrade_id, county_geoid):
    """ Get hourly timeseries for a combination of building id and an upgrade id

    The overall flow reproduces the Spark table created by
    https://github.com/rewiringamerica/pep/blob/dev/src/process/process_eulp_timeseries.py

    TODO: add file caching layer, to save money and speed up
        Parquet files are ~3Mb each, so full set for one upgrade is 500K * 3Mb = 1.5Tb
        GCP traffic in North America is ~8c per Gb, so, ~$150 per epoch without caching.
        Caching a reduced set of columns (~1/2) and resamapled hourly (1/4) will take
        500K * 3Mb / (2*4) = 200Gb per upgrade.

    TODO: test reading from a local machine and reading from NREL S3 instead of RA's GCS
        this function takes ~0.5s to execute on average, which is a bottleneck

    Args:
        building_id: (int) ResStock building ID
        upgrade_id: (int) ResStock upgrade ID
        county_geoid: (str) in.county from ResStock building metadata, e.g. 'G0101270'

    Returns:
        (pd.DataFrame): a dataframe with a sorted timestamp index and columns
            representing KWh consumption by fuel type and appliance
            Output column name format: {fuel type}.{appliance}
            fuel type: electricity, fuel_oil, natural gas, propane, site_energy
                (the last one is everything else combined)
            appliance: ceiling_fan, clothes_dryer, clothes_washer,
                cooling_fans_pumps, cooling, dishwasher, freezer,
                heating_fans, heating, heating_hp_bkup, hot_tub, hot_water,
                lighting_exterior, lighting_interior,
                lighting_garage, mech_vent, plug_loads, pool_heater, pool_pump,
                pv, range_oven, refrigerator, well_pump,
                fireplace, grill, total (all appliances of the same fuel type),
                net (including solar panels, pv)

    >>> get_hourly_outputs(100066, 0, 'G0101270').shape[0]
    8760
    >>> get_hourly_outputs(100066, 0, 'G0101270').shape[1] > 2  # 2+ groups
    1
    """
    state = get_state_code_from_county_geoid(county_geoid)
    pqpath = HOURLY_OUTPUT_PATH.format(
        building_id=building_id, upgrade_id=upgrade_id, state=state)
    # To save RAM, it'd be good to cache the columns of the dataset and read
    # only the needed ones. So, we need a stateful function - this is a dirty
    # hack to implement this.
    # TODO: abuse metaclasses instead?
    if not hasattr(get_hourly_outputs, 'columns'):
        pqtemp = pd.read_parquet(pqpath).sort_values('timestamp')
        # skipping intensity and emissions columns
        columns = [
            column for column in pqtemp.columns
            if column == 'timestamp' or column.endswith('.energy_consumption')
        ]
        setattr(get_hourly_outputs, 'columns', columns)
        column_renames = {
            col: col[4:-19] for col in columns
            if col.startswith('out.') and col.endswith('.energy_consumption')
        }
        setattr(get_hourly_outputs, 'column_renames', column_renames)
        timestep = pqtemp.iloc[1]['timestamp'] - pqtemp.iloc[0]['timestamp']
        setattr(get_hourly_outputs, 'timestep', timestep)
        fuel_types = set()
        appliance_types = set()
        appliance_groups = {}
        for column in column_renames.values():
            if '.' not in column:  # timestamp
                continue
            fuel_type, appliance = column.split('.', 1)
            fuel_types.add(fuel_type)
            appliance_types.add(appliance)
            appliance_groups.setdefault(appliance, [])
            appliance_groups[appliance].append(column)

        fuel_types -= {'site_energy'}
        # maybe remove appliances not covered by upgrades: grill, pool_pump,
        # and hot tub heater
        # fireplace in ResStock are only gas powered (i.e., not wood) and should
        # be counted towards heating
        appliance_types -= {'total', 'net', 'grill', 'pool_pump', }
        setattr(get_hourly_outputs, 'fuel_types', fuel_types)
        setattr(get_hourly_outputs, 'appliance_types', appliance_types)
        setattr(get_hourly_outputs, 'appliance_groups', appliance_groups)
        # appliance mapping to aggregate by purpose
        # TODO: make groups separable by fuel, e.g. backup heating should be
        # in a separate group, and heating fans should be separate, too.
        setattr(get_hourly_outputs, 'consumption_groups', {
            'heating': [
                'heating', 'heating_fans_pumps', 'heating_hp_bkup', 'fireplace',
            ],
            'cooling': ['cooling', 'cooling_fans_pumps', ],
            'lighting': [
                'lighting', 'lighting_interior', 'lighting_exterior',
                'lighting_garage',
            ],
            'other': [
                'hot_tub_heater', 'hot_tub_pump', 'hot_water', 'well_pump',
                'dishwasher',  'freezer', 'refrigerator', 'grill', 'range_oven',
                # should fans and mech vent  be considered cooling/heating?
                'ceiling_fan', 'mech_vent',
                'pool_heater', 'pool_pump',
                'clothes_dryer', 'clothes_washer',
                'plug_loads',  # pv,  # not considering solar (pv) yet
            ]
        })

    ho = (
        pd.read_parquet(pqpath, columns=get_hourly_outputs.columns)
        .set_index('timestamp')
        .sort_index()
    )

    # timestamps indicate the end of the period.
    # To make use of pandas resampling, they should be set at the start
    ho = (
        ho.set_index(ho.index-get_hourly_outputs.timestep)
        .rename(columns=get_hourly_outputs.column_renames)
    )
    # TODO: combine both aggregations
    ho = pd.DataFrame({
        appliance: ho[col_names].sum(axis=1)
        for appliance, col_names in get_hourly_outputs.appliance_groups.items()
    })
    ho = pd.DataFrame({
        group_name: ho[col_names].sum(axis=1)
        for group_name, col_names in get_hourly_outputs.consumption_groups.items()
    })

    return ho.resample('H').sum()


@file_cache(CACHE_PATH)
def get_weather_file(county_geoid):
    """ Retrieve weather timeseries for a given county geoid in ResStock

    It takes about 150..200ms to read a file from a GCP bucket. With ~3K files,
    that's ~10min worst case for the entire dataset. This function returns all
    columns - filtering out the right features is a task for the dataset
    generator.

    Args:
        county_geoid: (str) ResStock county geoid

    Returns:
        pd.DataFrame: A dataframe with 8760 rows (hours in a non-leap year) and
            columns `temp_air`, `relative_humidity`, `wind_speed`, `weekend`,
            `wind_direction`, `ghi`, `dni`, `diffuse_horizontal_illum`,

    >>> weather_df = get_weather_file('G0200130')
    >>> isinstance(weather_df, pd.DataFrame)
    True
    >>> weather_df.shape[0] == HOURS_IN_A_YEAR  # 8760
    True
    >>> weather_df.shape[1] >= 5
    True
    """
    state = get_state_code_from_county_geoid(county_geoid)
    weather_file_path = WEATHER_FILES_PATH.format(state=state, geoid=county_geoid)
    df = pd.read_csv(
        weather_file_path, parse_dates=['date_time'], index_col=['date_time']
    ).rename(columns={
        'Dry Bulb Temperature [°C]': 'temp_air',
        'Relative Humidity [%]': 'relative_humidity',
        'Wind Speed [m/s]': 'wind_speed',
        'Wind Direction [Deg]': 'wind_direction',
        'Global Horizontal Radiation [W/m2]': 'ghi',
        'Direct Normal Radiation [W/m2]': 'dni',
        'Diffuse Horizontal Radiation [W/m2]': 'diffuse_horizontal_illum'
    })
    # in TMY3 files, weather year is a combination of months from different
    # years. Resstock overrides year for these files, so only month-day-hour
    # portion matters
    df = df.assign(
        # Monday is 0
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.weekday.html
        weekend=df.index.weekday.isin((5, 6)).astype(int)
    ).set_index(df.index.strftime("%m-%d-%H:00")).sort_index()

    return df


def apply_upgrades(building_features, upgrade_id):
    # TODO: actually apply upgrades according to
    # https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf
    # upgrades are dependent on: IECC zone, wall type, etc;
    # so, full implementation will require changes to baseline columns provided
    # in `building_features`
    match upgrade_id:
        case 0:  # baseline
            return building_features
        # case 1: # basic enclosure
        # case 2: # enhanced enclosure
        # case 3: # heat pump, min efficiency, electric backup
        # case 4: # heat pump, high efficiency, electric backup
        # case 5: # heat pump, high efficiency, existing heating as backup
        # case 6: # heat pump water heater
        # case 7: # whole home electrification, min efficiency
        # case 8: # whole home electrification, high efficiency
        # case 9: # whole home electrification, high efficiency+basic enclosure
        case _:
            raise ValueError(r"Upgrade id={upgrade_id} is not yet supported")


def train_test_split(dataset: np.array, left_size):
    """ Split the provided array into two random shares

    Why: `tf.keras.utils.split_dataset()`-based iterators are a bit slow
    for small experiments, with iteration over 500k examples taking 25s.

    For larger datasets where records might not fit into workers RAM,
    `tf.keras.utils.split_dataset()` is still the preferred option
    """
    np.random.shuffle(dataset)
    split_point = int(len(dataset)*left_size)
    return dataset[:split_point], dataset[split_point:]


class DataGen(tf.keras.utils.Sequence):
    batch_size: int
    upgrades = (0,)  # upgrades to consider. 0 is the baseline scenario
    weather_features = ('temp_air', 'ghi', 'wind_speed', 'weekend')
    building_features = BuildingMetadataBuilder.supported_building_features
    consumption_groups = ('heating', 'cooling', 'lighting',)  # skip 'other'
    # column mapping to aggregate hourly outputs by appliance
    appliance_groups: Dict[str, str] = None
    time_granularity = None
    weather_files_cache: Dict[str, np.array]
    # Building ids only, not combined with upgrades.
    # Not used, for debugging purpose only
    building_ids: np.array
    # np.array() N x 2 with, the first column being a building id and the
    # second - upgrade id
    ids = None
    output_length: int = None

    def __init__(self, building_ids, upgrade_ids=None, weather_features=None,
                 building_features=None, consumption_groups=None,
                 time_granularity='Y', batch_size=64, metadata_builder=None):
        """
        Args:
            building_ids: (Iterable[int]) integer ids of the buildings in this
                data generator. This is usually some portion of the target
                population of ResStock buildings.
            upgrade_ids: (Iterable[int]) integer ids of supported upgrades.
                ResStock natively supports upgrades 0-10, with 0 being the
                baseline scenario.
            weather_features: (Iterable[str]) weather features to be taken into
                account. Features should be a subset of columns returned by
                `get_weather_file()`. Default is to use the same features as the
                simulation, so there is little reason to change these.
            building_features: (Iterable[str]) building features to use. These
                features should be a subset of columns returned by the
                `metadata_builder`
            consumption_groups: (Tuple[str]) appliance groups to be used, e.g.
                cooling, heating, etc. Groups should be a subset of columns
                returned by `get_hourly_outputs()`
            time_granularity: (str) level of timeseries aggregation, one of:
                - `H`: hourly
                - `D`: daily
                - `M`: monthly
                - `Q`: quarterly
                - `Y`: yearly
            batch_size: (int) self-explanatory.
            metadata_builder: (callable) a method to retrieve metadata for a
                single building given its int id in ResStock
        """
        self.upgrades = tuple(upgrade_ids or self.upgrades)
        self.weather_features = list(weather_features or self.weather_features)
        self.building_features = [
            feature for feature in (building_features or self.building_features)
            if feature != 'county'
        ]
        self.consumption_groups = list(consumption_groups or self.consumption_groups)
        self.time_granularity = time_granularity
        self.weather_files_cache = {}
        self.batch_size = batch_size
        self.building_ids = np.fromiter(building_ids, int)
        self.ids = np.array(list(itertools.product(self.building_ids, self.upgrades)))
        self.metadata_builder = metadata_builder or BuildingMetadataBuilder()
        # TODO: make this (specifically, __getitem__) work for variable length input
        self.output_length = {
            'H': HOURS_IN_A_YEAR,
            'D': 365,
            'M': 12,
            'Q': 4,
            'Y': 1,
        }[time_granularity]

    def __len__(self):
        # number of batches; last batch might be smaller
        return math.ceil(len(self.ids) / self.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.ids)

    def get_weather_data(self, county_geoid):
        """ In-memory caching of weather data """
        if county_geoid not in self.weather_files_cache:
            df = get_weather_file(county_geoid)
            self.weather_files_cache[county_geoid] = df[self.weather_features]
        return self.weather_files_cache[county_geoid]

    def __getitem__(self, idx):
        """ Generate a batch #`idx`

        This method should produce a dictionary of numpy arrays (or tensors)
        """
        batch_ids = self.ids[idx*self.batch_size:(idx+1)*self.batch_size]
        building_inputs = np.empty((self.batch_size, len(self.building_features)), dtype=np.float16)
        # keras convolutions only support NHWC (i.e., channels last)
        # so, time dimension comes first, than features
        weather_inputs = np.empty((self.batch_size, HOURS_IN_A_YEAR, len(self.weather_features)), dtype=np.float16)
        outputs = np.empty((self.batch_size, len(self.consumption_groups), self.output_length), dtype=np.float16)

        for i, (building_id, upgrade_id) in enumerate(batch_ids):
            building_features = self.metadata_builder(building_id).copy()
            building_features = apply_upgrades(building_features, upgrade_id)
            # ~0.1ms per building up to this point
            county_geoid = building_features['county']
            # limit to features used in this experiment
            # 0.6ms up to here
            building_inputs[i] = building_features[self.building_features].values

            weather_inputs[i] = self.get_weather_data(county_geoid).values

            outputs[i] = get_hourly_outputs(
                building_id, upgrade_id, county_geoid
            )[self.consumption_groups].resample(self.time_granularity).sum().T

        # TODO: maybe split into distinct features and concat in the model?
        return {
            'building_features': building_inputs,
            'weather_data': weather_inputs
        }, {
            'outputs': outputs
        }
