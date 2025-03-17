# Features & Upgrades

## Features

The following list describes mapping from Simulation Features to Surrogate Model Features:

**Simulation Features:** The minimum set of housing characteristics that can completely specify the simulation inputs. For each characteristic that is an input to the EnergyPlus simulation: if it satisfies the inclusion condition (non-deterministic OR has more than one parent), we include it in the set; otherwise, we find the first ancestor that satisfies the condition.

**Surrogate Model Features**: Simulation inputs transformed for the surrogate model. The [options_lookup.tsv](https://github.com/NREL/resstock/blob/run/euss/resources/options_lookup.tsv) defines the mapping from select ResStock characteristics to EnergyPlus arguments. In many cases this provides a mapping from a categorical feature to a set of numerical features, and so we will likely want to leverage this mapping. In future work, we will directly use this .tsv file to apply the mapping, but for the MVP we decided to just define these manually and non-comprehensively.

We consider the set of Simulation Features, the details for which can be viewed in this sheet when applying the “Simulation Features” filtered view. Note that this list of features is based on ResStock 2024.2 features that are relevant for the scope of buildings supported by the model: Occupied homes without shared HVAC or water heating systems or un-modeled fuels (e.g, wood or coal) and that fall into the following housing categories: single family homes (attached and detached), mobile homes, and multi-family homes with <5 units. Because we train the model on a mix samples from 2024.2 and 2022.1, we have to map 2022.1 features to 2024.2, [see details below](#aligning-2022-to-2024-metadata). 


### Simulation Features → Surrogate Model Features

The mapping takes the following format:

> Simulation Feature:
>
> `surrogate_model_feature` (dtype) : description of feature and transformation

**Feature List**:

ASHRAE IECC Climate Zone 2004:

`climate_zone_temp` (int): The first character of the climate zone category, representing the temperature zone as a number.

`climate_zone_moisture` (str): The second character of the climate zone category, representing the moisture zone.

Bedrooms:

`n_bedrooms` (int): Number of bedrooms in the housing unit.

`n_bathrooms` (float): Number of bathrooms in the housing unit calculated as `n_bedrooms` / 2 + .5 (see [Building America docs p21](https://www.nrel.gov/docs/fy10osti/47246.pdf)). While this is clearly a deterministic function of `n_bedrooms`, we include it here because (1) it is an input for calculating water heater capacity, and (2) we’ll want to potentially plug this in as a known value at inference time.

Ceiling Fan:

`has_ceiling_fan` (bool): Indicator for whether a ceiling fan is in the unit. This simulation feature only takes on two values, one of which is `None`, so we decided to code it as an indicator.

Clothes Dryer:

`clothes_dryer_fuel` (str): Fuel type of the clothes dryer. 

Clothes Washer: 

`clothes_washer_efficiency` (str): Type of the clothes washer (i.e, 'EnergyStar', 'Standard', or 'None')

Cooking Range:

`cooking_range_fuel` (str): Fuel type of the cooking range.

Cooling Setpoint:

`cooling_setpoint_degrees` (double): Cooling setpoint in Fahrenheit.

Cooling Setpoint Offset Magnitude:

`cooling_setpoint_offset_magnitude_degrees_f` (double): Number of degrees (F) offset from the cooling setpoint for the cooling setback.

Dishwasher:

`dishwasher_efficiency_kwh` (int): Rated efficiency of dishwasher in kWh, set to 9999 (infinite efficiency) if not present.

Duct Leakage and Insulation:

`duct_insulation_r_value` (int): Duct insulation r-value, set to 0 if uninsulated, and to 999 (infinite efficiency) if not present.

`duct_leakage_percentage` (double) : Duct leakage percentage, set to 0 if ducts are not present. Percentage is divided by 100.

Duct Location:

`duct_location` (bool): Indicator for whether the unit has ducts. TODO: add this features

Geometry Attic Type:

`attic_type` (str): Type of attic (e.g., “None”, “Vented Attic”)

Geometry Building Horizontal Location SFA / Geometry Building Horizontal Location MF

`is_middle_unit` (bool): Indicator for whether the unit is a middle unit in the building. In the simulation feature, there are 3 options for attached homes: left, right and middle. Since middle unit seems like the only thing that would affect energy consumption, this is coded as a binary indicator for whether or not it is a middle unit, which is always false for detached home.

Geometry Building Level MF

`unit_level_in_building` : The level of the unit within the building: `Top`, `Middle`, `Bottom` , for Multi-Family, otherwise `None`.

Geometry Building Number Units SFA / Geometry Building Number Units MF

`n_building_units` (int): Number of units in the building, set to 1 for all detached homes and mobile homes.

Geometry Building Type ACS:

`is_attached` (bool): Indicator for whether the home is attached (includes single family attached and multi-family).

`is_mobile_home` (bool): Indicator for whether the home is a mobile home.

Geometry Floor Area: *We actually will use the sqft simulation feature which is created in EnergyPlus pre-processing*

`sqft` (double): Finished floor area of the housing unit in ft^2. We actually directly use the `sqft`, which is derived from the Geometry Floor Area variable, which is the average conditioned floor area of AHS surveyed homes within the given housing type and Geometry Floor Area Bin. This variable is produced by a preprocessing step in EnergyPlus and output, but it isn’t technically sampled metadata feature, which is why it isn’t in this list.

Geometry Foundation Type

`foundation_type` : Type of foundation (e.g, “Ambient”, “Heated Basement”).

Geometry Garage:

`garage_size_n_car` (int): Size of the garage measured in the number of cars it holds, set to 0 if no garage is present.

Geometry Stories:

`n_stories` (int): Number of stories of the building.

Geometry Wall Exterior Finish:

`exterior_wall_material` (str): Material of the exterior wall (e.g, “Aluminum” , “Brick”)

`exterior_wall_color` (str) : Color of the exterior wall (e.g, “Light” , “Medium/Dark”).

Heating Fuel:

`heating_fuel` (str): Heating fuel.

Heating Setpoint:

`heating_setpoint_degrees` (double): Heating setpoint in Fahrenheit.

Heating Setpoint Offset Magnitude

`heating_setpoint_offset_magnitude_degrees_f` (double): Number of degrees (F) offset from heating setpoint for the heating setback.

HVAC Cooling Efficiency:

`ac_type` (str): Type of cooling system (e.g., “AC”, “Room AC”).

`cooling_efficiency_eer` (double): Energy efficiency ratio (EER) of the cooling system, set to 999 (infinite efficiency) if not present. Note that when the ac type is a heat pump, this is instead derived from the “HVAC Heating Efficiency” feature.

HVAC Cooling Partial Space Conditioning:

`cooled_space_percentage` (double): Percentage of living space that is air conditioned. Percentage is divided by 100.

HVAC Has Ducts:

`has_ducts` (bool): Indicator for whether the unit has ducts. Note that technically this is a deterministic feature of Duct Location, but since Duct Location was not present in 2022.1, we keep this slightly redudant feature just in case. 

HVAC Heating Efficiency:

`heating_efficiency_nominal_percent` (double): Nominal efficiency percentage of the heating system (i.e. percentage of energy consumed by the system that is actually converted to useful heat), set to 900% (infinite efficiency) if not present (see [conversion formulas](https://www.energyguru.com/EnergyEfficiencyInformation.htm)). Percentage is divided by 100.

`heating_appliance_type`(str): Type of heating appliance (e.g., “Furnace”, “Boiler”).

HVAC Heating Type:

`has_ductless_heating` (bool):  Indicator for whether the heating is ductless. This simulation feature contains information on whether or not the heating system is a heat pump, but this is already captured by `heating_appliance_type` . Importantly, this variable is not fully captured by `has_ducts` since there are units have have ducts but still have a ductless heating system.

Infiltration:

`infiltration_ach50` (int): Air leakage rates for the living and garage spaces in air changes per hour at 50 pascals (ACH50).

Insulation Ceiling

`insulation_ceiling_r_value` (int): R-value of the foundation ceiling, set to 0 if not present or or uninsulated.

Insulation Floor:

`insulation_floor_r_value` (int): R-value of the floor, set to 0 if not present or uninsulated.

Insulation Foundation Wall

`insulation_foundation_wall_r_value` (int): R-value of the foundation walls, set to 0 if not present or or uninsulated.

Insulation Roof

`insulation_roof_r_value` (int): R-value of the foundation ceiling, set to 0 if not present or or uninsulated.

Insulation Slab

`insulation_slab_r_value` (int): R-value of the slab, set to 0 if not present or uninsulated.

Insulation Wall

`insulation_wall_r_value` (int): R-value of the walls, set to 0 if not present or uninsulated.

Lighting:

`lighting_efficiency` (double) : Luminous efficiency of the light type, expressed as a a percentage using [these conversions](https://en.wikipedia.org/wiki/Luminous_efficacy#Lighting_efficiency). Percentage is divided by 100.

Misc Extra Refrigerator:

`refrigerator_extra_efficiency_ef` (double): Energy factor (EF) of the refrigerator.

Misc Freezer:

`has_standalone_freezer` (bool): Indicator for whether a standalone freezer is present. This is coded as an indicator because the simulation feature has only non-None possible value.

Misc Gas Fireplace:

`has_gas_fireplace` (bool): Indicator for whether the unit has a gas fireplace. This is coded as an indicator because the simulation feature has only one non-None possible value.

Misc Gas Grill:

`has_gas_grill` (bool): Indicator for whether the unit has a gas grill. This is coded as an indicator because the simulation feature has only one non-None possible value.

Misc Gas Lighting:

`has_gas_lighting` (bool): Indicator for whether the unit has gas lighting. This is coded as an indicator because the simulation feature has only one non-None possible value.

Misc Hot Tub Spa:

`hot_tub_spa_fuel` (str): Fuel type of the hot tub.

Misc Pool Heater:

`pool_heater_fuel` (str): Fuel type of the pool heater.

Misc Well Pump:

`has_well_pump` (bool): Indicator for whether the unit has a well pump. This is coded as an indicator because the simulation feature has only one non-None possible value.

Neighbors:

`neighbor_distance_ft` (float): Distance between the unit and the nearest neighbors to the left and right in feet, coded as 15 for “Left/Right at 15ft”, and 9999 (infinite distance) if no neighbors are present.

Occupants:

`n_occupants` (int): Number of occupants in the dwelling unit, with “10+” coded as 11.

Orientation:

`orientation_degrees` (int): Compass orientation of the unit translated to degrees, with N represented as 0.

Plug Loads:

`plug_load_percentage` (double): Plug load usage level as a percentage, with 100% being the median possible value. Percentage is divided by 100.

Refrigerator:

`refrigerator_efficiency_ef` (double): Energy factor (EF) of the refrigerator. This simulation feature contains both EF and usage percentage, but the usage percentage is always 100%, so it is not used.

Roof Material:

`roof_material` (str): Roof material type (e.g, “Slate”, “Asphalt Shingles, Medium”).

Usage Level:

`usage_level_appliances` (int): Usage-level of major appliances relative to the national average, with “Low”, “Medium” and “High” represented as integers 1, 2, and 3 respectively. This single usage level applies is applied to the following appliances: clothes dryer, clothes washer, cooking range, dishwasher, hot water fixtures, plug load diversity, refrigerator.

Vintage:

`vintage` (int): Vintage of the building, with “<1940” coded as 1930.

Water Heater Efficiency:

Water Heater Efficiency is a single string containing the fuel, type, and in some cases, tank volume and efficiency. This is mapped to various specs using based on the `options.tsv`.

`water_heater_fuel` (str): Fuel type of the water heater.

`water_heater_type` (str): Type of water heater: “Storage”, "Instantaneous”, or “Heat Pump”.

`water_heater_tank_volume` (int): Capacity of the water heater tank in gallons, which is set to 0 for instantaneous (i.e., tankless water heaters). If not specified (only specified for HPWHs), it is calculated based on the number of bedrooms, bathrooms and whether the water heater is electric (see [Building America docs p12](https://www.nrel.gov/docs/fy10osti/47246.pdf)). According to the ResStock docs, this heuristic aligns with EnergyPlus preprocessing.

`water_heater_efficiency_ef` (double): Efficiency factor (EF) of the  of the water heater, which is the ratio of the useful energy output to the total amount of energy delivered to the water heater. If UEF is is provided (only ever used for HPWHs), it is converted to EF using the equation EF = 1.2101 * UEF - 0.6052 (see [RESSNET conversion tool](https://www.resnet.us/wp-content/uploads/RESNET-EF-Calculator-2017.xlsx)).

`water_heater_recovery_efficiency_ef` (double): Recovery efficiency factor (EF) of the water heater, which is the efficiency of heat of transferring heat fro the energy source to the water. While the options.tsv sets this to 0 for all electric or tankless water heaters since this is essentially ignored by the simulation, we set it to 1 which is a more appropriate value.


Water Heater Location: 

`water_heater_location` (str): Description of where in the unit the water heater is located, with 'Outside' if the water heater is not in the unit. TODO: add feature

Water Heater In Unit:

`has_water_heater_in_unit` (bool): Indicator for whether the water heater is in the unit. Note that this is deterministic function of Water Heater Location, but since this was not a feature in 2022.1, we keep this redudant feature just in case. 

Window Areas:

`window_to_wall_ratio` (double): Mean window to wall ratio (WWR) of the front, back, left, and right walls. This is translated from a string format that lists the WWR for each of the four walls.

Windows:

`window_ufactor` (double): The U-factor of the windows, which measure how well the window insulates, with lower values indicating better insulation

`window_shgc` (double): The Solar Heat Gain Coefficient (SHGC) in [0, 1] ,which measures how much of the sun’s heat comes through the window.

### Excluded Features

Bathroom Spot Vent Hour: Temporal variable seems unnecessary for predicting annual outputs.

Clothes Washer Presence: Deterministic function of `clothes_washer_efficiency`. 

Cooling Setpoint Has Offset: Deterministic function of `cooling_setpoint`.

Cooling Setpoint Offset Period: Temporal variable seems unnecessary for predicting annual outputs.

County and PUMA: while technically this is an input to the EnergyPlus, this is not actually used in the simulation— it is just for record-keeping/filtering/filenaming. 

Ground Thermal Conductivity: ****Not in 2022 dataset

Has PV: only has effect on pv outputs which we are not using

Heating Setpoint Has Offset: Deterministic function of `heating_setpoint`.

Heating Setpoint Offset Period: Temporal variable seems unnecessary for predicting annual outputs.

Misc Pool: Completely determined by `pool_heater_fuel`

PV Orientation: only relevant for PV outputs which are not being used

PV System Size: only relevant for PV outputs which are not being used

Range Spot Vent Hour: Temporal variable seems unnecessary for predicting annual outputs.

### Additional Features

Fuel type indicator columns: which may help the model predict specific fuel consumptions more accurately. While the various appliance fuel columns will be one hot encoded , the model would have to learn the relationships between one hot encodings and targets: e.g, one hot encoding for  `heating_fuel = 'Methane Gas'` OHE for`clothes_dryer_fuel = 'Methane Gas'` , `has_gas_grill=True` and the target `methane gas`. This also just helps facilitate post-processing, wherein a fuel prediction is set to 0 if no appliance uses that fuel. 

`has_methane_gas_appliance`

`has_fuel_oil_appliance`

`has_propane_appliance`

Heat Pump tech upgrade indicator columns: while these columns are not used for any of the baseline appliances, we need a way to distinguish these heat pump appliances from electric resistance in the upgrade metadata. Rather than just adding another category to `cooking_range_fuel` and `clothes_dryer_fuel`, we chose to represent this as a boolean indicator so that electric resistance and heat pumps would still have a common feature value, which may be useful for predicting the `electricity` target. 

`has_heat_pump_dryer`

`has_induction_range`

Additional Heat Pump Detail Columns: The heat pump sizing method is specified in the options.tsv rather than in the samples, so since we are altering this methodology in some of the RAStock HP upgrades, we need a column to reflect this. 

`heat_pump_sizing_methodology` (str): The methodology used for sizing the heat pump, which is either `ACCA` (default) `HERS` (for upgrade 11.05) or `None` (if no heat pump).

### Aligning 2022 to 2024 metadata

| **Simulation Feature**                     | **2024 Dataset**                                      | **2022 Dataset**                                      | **Notes** |
|----------------------------------|------------------------------------------------------|------------------------------------------------------|----------|
| Appliance Features: `clothes_dryer`, `clothes_washer`,`cooking_range`, `dishwasher`,`refrigerator` | Appliance type and usage split into separate columns, e.g., `clothes_dryer`, `clothes_dryer_usage` | Column contained both appliance type and usage details | Usage percentage are entirely determined by the `usage_level`, so in 2022 we simply remove this extra usage detail string and in 2024 we drop the specific appliance usage columns, since this is deterministic|
| Ducted vs. ductless heat pump naming terminology in `hvac_heating_type_and_fuel`, `hvac_heating_efficiency`,  `hvac_cooling_efficiency`, `hvac_cooling_type`| The columns now specify `"MSHP"` (minisplit) vs `"ASHP"` for heating depending on `hvac_has_ducts` and `"Ducted Heat Pump"` vs `"Non-Ducted Heat Pump"` depending on `hvac_has_ducts`. | There were no ductless heat pumps modeled in baseline, so `"ASHP"` or `"Heat Pump"` was used for all these columns without specifying ducted. | Aligned mostly to 2022, since this distinction is entirely determined by `hvac_has_ducts`. However in the case of `hvac_cooling_type`, we do specify `"Ducted Heat Pump"` vs `"Non-Ducted Heat Pump"` because this aligns with the terminology for `hvac_heating_type` in both 2022 and 2024, which had the options: `"Ducted Heating"`,`"Non-Ducted Heating"`,  `"Ducted Heat Pump"`|
| `geometry_attic_type` | Possible values include `"Unvented Attic"` | | Aligned to 2024 without taking action: this is just a new category value  |
| `duct_leakage_and_insulation` | | Column called just `ducts` and strings do not contain specific detail of "Leakage to Outside", just says "Leakage"| Aligned to 2024 by simply adding the "Leakage to Outside" string |
| `duct_location`           | Provided specific locations (Attic, Crawlspace, Garage, etc.) | This column did not exist   | Aligned mostly 2024 by mapping cases where `hvac_has_ducts="No"` to `"None"`and where `hvac_has_ducts="Yes"` to `"Unknown Location"`, which is a new token  |
| `water_heater_location`       | Provides specific locations of water heater (Attic, Crawlspace, Garage, Heated Basement, etc.) | This column did not exist  | Aligned mostly to 2024 by mapping cases where `water_heater_in_unit="No"` to `"None"`and where `water_heater_in_unit="Yes"` to `"Inside Unit"`, which is a new token |
| `misc_hot_tub_spa` | Possible Values: `"Electricity"`, `"Natural Gas"`, `"Other Fuel"`     | Possible Values: `"Electric"`, `"Gas"`                              | Aligned to 2024 by updating naming conventions |
| `misc_pool_heater` |Possible Values:  `"Electricity"`, `"Natural Gas"`, `"Other Fuel"`, `"None"`     |Possible Values: `"Electric"`, `"Gas"`, `"Solar"`, `"None"`    | Aligned to 2024 by updating naming conventions and mapping `"Solar"` in 2022 to `"Other Fuel"` |
| `cooking_range` | Possible Values include `"Electric Resistance"`, `"Electric Induction"`, `"Gas"` | Possible Values: `"Electric"`, `"Gas"`                              | Aligned to 2024 by renaming `"Electric"` to `"Electric Resistance"` |
| `refrigerator` | Possible values include`"EF 21.9"` added                                   | | Aligned to 2024 without taking action: this is just a new numerical value  |
| `water_heater_efficiency` | Only HPWH category is  `"Electric Heat Pump, 50 gal, 3.45 UEF"`            |  Only HPWH category is `"Electric Heat Pump, 80 gal"`                   | No action taken: the feature transformation will handle both and these get mapped to numerical features so both are fine |
| `county`, `county_and_puma` | Slightly different set of counties | Slightly different set of counties    | No action taken-- this is not transformed into a feature so misalignment here is fine |

Note that there are other demographic features (not used in the surrogate model) that are included in 2024 but not 2022 that we retain and these will just be null. Further there are a few deterministic features where the schemas are misaligned and these are not described here either since deterministic simulation features do not get convered into surrogate model features. 

## Upgrades

### Implementation
The upgrades are implemented by modifying the baseline surrogate model features to match the upgrade specification. This upgrade logic is determined by consulting the [EUSS Round 1 Technical Documentation and Measure Applicability Logic](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf), as well as the [upgrade yaml file](https://github.com/NREL/resstock/blob/run/euss/EUSS-project-file_2018_550k.yml) that is used to define the upgrades is ResStock precisely by defining this kind of feature update logic. Note most cases, the YAML is much more specific, and covers many edge cases not described in the technical documentation.

- Example of how to parse the YAML
    
    The following excerpt from the YAML file
    
    > upgrade_name: Basic Enclosure
    #Up01
    options:
    #Attic floor insulation up to IECC 2021 levels for homes with vented attics and less than R-30 existing insulation
    > 
    > - &attic_insulation_IECC_CZ1A
    > option: Insulation Ceiling|R-30
    > apply_logic:
    >     - and:
    >         - ASHRAE IECC Climate Zone 2004|1A
    >         - Geometry Attic Type|Vented Attic
    >         - or:
    >             - Insulation Ceiling|Uninsulated
    >             - Insulation Ceiling|R-7
    >             - Insulation Ceiling|R-13
    
    is equivalent to the following psuedocode:
    
    ```python
    #condition = attic_insulation_IECC_CZ1A
    if "ASHRAE IECC Climate Zone 2004" = "1A" and "Geometry Attic Type" = "Vented Attic" and "Insulation Ceiling" in ["Uninsulated", "R-7", "R-13"]:
    	"Insulation Ceiling" = "R-30"
    ```
    
### Upgrade List
We have implemented the logic for the following upgrades:

**ResStock EUSS 2022.1** (see [docs](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/EUSS_ResRound1_Technical_Documentation.pdf) for details):

* 0: Baseline

* 1: Basic Enclosure

* 3: Min-Efficiency Heat Pumps with Electric Backup

* 4: High-Efficiency Heat Pumps with Electric Backup

* 6: Heat Pump Water Heaters

* 9: High Efficiency Whole-Home Electrification + Basic Enclosure Package (1 + 4 + 6 + heat pump dryers + induction ranges)

**RAStock (Simulated by RA)**

* 11.05: Medium-Efficiency Heat Pumps (SEER 18, 10 HSPF HERS Sizing) with Electric Backup, No Setpoint Setback
    
    * Details: HVAC Heating Efficiency = ‘ASHP, SEER 18, 10 HSPF' if HVAC Has Ducts, otherwise 'ASHP, SEER 18, 10.5 HSPF’. This then gets transformed as described [here](https://www.notion.so/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=21). 
    
        `heat_pump_sizing_methodology`=HERS
        
        `cooling_setpoint_offset_magnitude_degrees_f` = 0
        
        `heating_setpoint_offset_magnitude_degrees_f` = 0
        

* 13.01: Medium-Efficiency Heat Pumps (SEER 18, 10 HSPF HERS Sizing) with Electric Backup, No Setpoint Setback + Basic Enclosure Package (1 + 11.05)

Note that while we have implemented the logic for heat pump dryers and induction ranges as standalone upgrades, we have chosen to not use these in training due to terrible performance on predicting these tiny savings values, particularly compared to the good performance of the benchmark on these low variance end uses.