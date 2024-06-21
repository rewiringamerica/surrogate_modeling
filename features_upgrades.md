# Features & Upgrades

## Features

The following list describes mapping from Simulation Features to Surrogate Model Features:

**Simulation Features:** The minimum set of housing characteristics that can completely specify the simulation inputs. For each characteristic that is an input to the EnergyPlus simulation: if it satisfies the inclusion condition (non-deterministic OR has more than one parent), we include it in the set; otherwise, we find the first ancestor that satisfies the condition.

**Surrogate Model Features**: Simulation inputs transformed for the surrogate model. The [options_lookup.tsv](https://github.com/NREL/resstock/blob/run/euss/resources/options_lookup.tsv) defines the mapping from select ResStock characteristics to EnergyPlus arguments. In many cases this provides a mapping from a categorical feature to a set of numerical features, and so we will likely want to leverage this mapping. In future work, we will directly use this .tsv file to apply the mapping, but for the MVP we decided to just define these manually and non-comprehensively.

The mapping takes the following format:

> Simulation Feature:
>
> `surrogate_model_feature` (dtype) : description of feature and transformation

### Simulation Features → Surrogate Model Features

ASHRAE IECC Climate Zone 2004:

`climate_zone_temp` (int): The first character of the climate zone category, representing the temperature zone as a number.

`climate_zone_moisture` (str): The second character of the climate zone category, representing the moisture zone.

Bedrooms:

`n_bedrooms` (int): Number of bedrooms in the housing unit.

`n_bathrooms` (float): Number of bathrooms in the housing unit calculated as `n_bedrooms` / 2 + .5 (see [Building America docs p21](https://www.nrel.gov/docs/fy10osti/47246.pdf)). While this is clearly a deterministic function of `n_bedrooms`, we include it here because (1) it is an input for calculating water heater capacity, and (2) we’ll want to potentially plug this in as a known value at inference time.

Ceiling Fan:

`has_ceiling_fan` (bool): Indicator for whether a ceiling fan is the unit. This simulation feature only takes on two values, one of which is `None` , so we decided to code it as an indicator.

Clothes Dryer: _Note in ResStock 2024, the usage portion of this feature is split into a new variable._

`clothes_dryer_fuel` (str): Fuel type of the clothes dryer. This simulation feature contains presence, fuel and usage percentage, but we just extract fuel and presence since usage percentage for appliances is completely determined by “Usage Level”.

Clothes Washer: _Note in ResStock 2024, the usage portion of this feature is split into a new variable._

`clothes_washer_efficiency` (str): Fuel type of the clothes washer. This simulation feature includes presence, efficiency, and usage percentage, but we just extract the efficiency and presence to create a 3 level category, since usage percentage for appliances is completely determined by “Usage Level”.

Cooking Range: _Note in ResStock 2024, the usage portion of this feature is split into a new variable._

`cooking_range_fuel` (str): Fuel type of the cooking range. This simulation feature includes both fuel and usage percentage, but we just extract fuel since usage percentage for appliances is completely determined by “Usage Level”.

Cooling Setpoint:

`cooling_setpoint_degrees` (double): Cooling setpoint in Fahrenheit.

Cooling Setpoint Offset Magnitude:

`cooling_setpoint_offset_magnitude_degrees_f` (double): Number of degrees (F) offset from the cooling setpoint for the cooling setback.

Dishwasher:

`dishwasher_efficiency_kwh` (int): Rated efficiency of dishwasher in kWh, set to 9999 (infinite efficiency) if not present.

Duct Leakage and Insulation:

`duct_insulation_r_value` (int): Duct insulation r-value, set to 0 if uninsulated, and to 999 (infinite efficiency) if not present.

`duct_leakage_percentage` (double) : Duct leakage percentage, set to 0 if ducts are not present. Percentage is divided by 100.

HVAC Has Ducts: _Note that in ResStock 2024 this is expanded to the more descriptive “Duct Location” feature._

`has_ducts` (bool): Indicator for whether the unit has ducts.

Geometry Attic Type:

`attic_type` (str): Type of attic (e.g., “None”, “Vented Attic”)

Geometry Building Horizontal Location SFA

`is_middle_unit` (bool): Indicator for whether the unit is a middle unit in the building. In the simulation feature, there are 3 options for attached homes: left, right and middle. Since middle unit seems like the only thing that would affect energy consumption, this is coded as a binary indicator for whether or not it is a middle unit, which is always false for detached home.

Geometry Building Number Units SFA

`n_building_units` (int): Number of units in the building, set to 1 for all detached homes.

Geometry Building Type ACS:

`is_attached` (bool): Indicator for whether the single family home is attached.

Geometry Floor Area: _We actually will use the sqft simulation feature which is created in EnergyPlus pre-processing_

`sqft` (double): Finished floor area of the housing unit in ft^2. This is actually derived directly from the `sqft` variable which is the midpoint of the Geometry Floor Area variable. This variable is produced by a preprocessing step in EnergyPlus and output, but it isn’t technically sampled metadata feature, which is why it isn’t in this list.

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

`cooling_efficiency_eer` (double): Energy efficiency ratio (EER) of the cooling system, set to 999 (infinite efficiency) if not present. Note that when the ac type is a heat pump, this is instead derived from the “HVAC Heating Efficiency” feature. If the unit has a heat pump for cooling and has “Shared Heating”, then there is no way to get the efficiency, so we just use the efficiency for Shared Cooling provided in the options.tsv.

HVAC Cooling Partial Space Conditioning:

`cooled_space_percentage` (double): Percentage of living space that is air conditioned. Percentage is divided by 100.

HVAC Heating Efficiency:

`heating_efficiency_nominal_percent` (double): Nominal efficiency percentage of the heating system (i.e. percentage of energy consumed by the system that is actually converted to useful heat), set to 900% (infinite efficiency) if not present (see [conversion formulas](https://www.energyguru.com/EnergyEfficiencyInformation.htm)). Percentage is divided by 100.

`heating_appliance_type`(str): Type of heating appliance (e.g., “Furnace”, “Boiler”).

HVAC Heating Type:

`has_ductless_heating` (bool): Indicator for whether the heating is ductless. This simulation feature contains information on whether or not the heating system is a heat pump, but this is already captured by `heating_appliance_type` . Importantly, this variable is not fully captured by `has_ducts` since there are units have have ducts but still have a ductless heating system.

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

`has_gas_fireplace` (bool): Indicator for whether the unit has a gas fireplace. This is coded as an indicator because the simulation feature has only non-None possible value.

Misc Gas Grill:

`has_gas_grill` (bool): Indicator for whether the unit has a gas grill. This is coded as an indicator because the simulation feature has only non-None possible value.

Misc Gas Lighting:

`has_gas_lighting` (bool): Indicator for whether the unit has gas lighting. This is coded as an indicator because the simulation feature has only non-None possible value.

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

`orientation_degrees` (int): Compass orientation of the unit translated to degrees, with North represented as 0, Northeast as 45, etc.

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

Water Heater Efficiency is a single string containing the fuel, type, and in some cases, tank volume and efficiency. This is mapped to various specs using based on the options.tsv.

`water_heater_fuel` (str): Fuel type of the water heater.

`water_heater_type` (str): Type of water heater: “Storage”, "Instantaneous”, or “Heat Pump”.

`water_heater_tank_volume` (int): Capacity of the water heater tank in gallons, which is set to 0 for instantaneous (i.e., tankless water heaters). If not specified (only specified for HPWHs), it is calculated based on the number of bedrooms, bathrooms and whether the water heater is electric (see [Building America docs p12](https://www.nrel.gov/docs/fy10osti/47246.pdf)). According to the ResStock docs, this heuristic aligns with EnergyPlus preprocessing.

`water_heater_efficiency_ef` (double): Efficiency factor (EF) of the of the water heater, which is the ratio of the useful energy output to the total amount of energy delivered to the water heater. If UEF is is provided (only ever used for HPWHs), it is converted to EF using the equation EF = 1.2101 \* UEF - 0.6052 (see [RESSNET conversion tool](https://www.resnet.us/wp-content/uploads/RESNET-EF-Calculator-2017.xlsx)).

`water_heater_recovery_efficiency_ef` (double): Recovery efficiency factor (EF) of the water heater, which is the efficiency of heat of transferring heat fro the energy source to the water. While the options.tsv sets this to 0 for all electric or tankless water heaters since this is essentially ignored by the simulation, we set it to 1 which is a more appropriate value.

Water Heater In Unit: _In 2024 ResStock this is a deterministic function of Water Heater Location, but the latter is not a feature in 2022_

`has_water_heater_in_unit` (bool): Indicator for whether the water heater is in the unit.

Window Areas:

`window_to_wall_ratio` (double): Mean window to wall ratio (WWR) of the front, back, left, and right walls. This is translated from a string format that lists the WWR for each of the four walls.

Windows:

`window_ufactor` (double): The U-factor of the windows, which measure how well the window insulates, with lower values indicating better insulation

`window_shgc` (double): The Solar Heat Gain Coefficient (SHGC) in [0, 1] ,which measures how much of the sun’s heat comes through the window.

### Additional Features

Fuel type indicator columns: which may help the model predict specific fuel consumptions more accurately. While the various appliance fuel columns will be one hot encoded , the model would have to learn the relationships between one hot encodings and targets: e.g, one hot encoding for  `heating_fuel = 'Methane Gas'` OHE for`clothes_dryer_fuel = 'Methane Gas'` , `has_gas_grill=True` and the target `methane gas`. This also just helps facilitate post-processing, wherein a fuel prediction is set to 0 if no appliance uses that fuel. 

`has_methane_gas_appliance`

`has_fuel_oil_appliance`

`has_propane_appliance`

Heat Pump tech upgrade indicator columns: while these columns are not used for any of the baseline appliances, we need a way to distinguish these heat pump appliances from electric resistance in the upgrade metadata. Rather than just adding another category to `cooking_range_fuel` and `clothes_dryer_fuel`, we chose to represent this as a boolean indicator so that electric resistance and heat pumps would still have a common feature value, which may be useful for predicting the `electricity` target. 

`has_heat_pump_dryer`

`has_induction_range`

## Upgrades

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
    

We have implemented the logic for upgrades:

0. Baseline

1. Basic Enclosure

3. Heat Pumps, Min-Efficiency, Electric Backup 

4. Heat Pumps, High-Efficiency, Electric Backup 

6. Heat Pump Water Heaters 

9.  Whole-Home Electrification, High Efficiency + Basic Enclosure Package (1 + 4 + 6 + heat pump dryers + induction ranges)

Note that while we have implemented the logic for heat pump dryers and induction ranges as standalone upgrades, we have chosen to not use these in training due to terrible performance on predicting these tiny savings values, particularly compared to the good performance of the benchmark on these low variance end uses.