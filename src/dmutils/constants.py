# TODO: delete this file once dmutils is updated
KILOWATT_HOUR_TO_BRITISH_THERMAL_UNIT = 3412.14
"""
The number of kilowatt-hours in British thermal units (BTU)
"""

BRITISH_THERMAL_UNIT_TO_KILOWATT_HOUR = 1 / KILOWATT_HOUR_TO_BRITISH_THERMAL_UNIT
"""
The number British thermal units (BTU) in a kilowatt-hour
"""

POUND_TO_KILOGRAM = 0.453592
"""
The number of pounds in a kilogram.
"""

KILOGRAM_TO_POUND = 1 / POUND_TO_KILOGRAM
"""
The number of kilograms in pound.
"""

SEER_TO_EER = 0.875
"""
Conversion of SEER (Seasonal Energy Efficiency Ratio) to EER (Energy Efficiency Ratio).

See https://energy-models.com/tools/how-convert-seer-eer-or-cop-kwton-hspf
"""

EER_TO_SEER = 1 / SEER_TO_EER
"""
Conversion of EER (Energy Efficiency Ratio) to SEER (Seasonal EER).

See https://energy-models.com/tools/how-convert-seer-eer-or-cop-kwton-hspf
"""

EER2_TO_EER = 1.04
"""
Conversion of SEER2 to SEER (Seasonal Energy Efficiency Ratio)
for Packaged Air Conditioner and Heat Pump.
The same conversion can be used for SEER2 to SEER.

See https://www.marathonhvac.com/seer-to-seer2
"""

EER_TO_EER2 = 1 / EER2_TO_EER
"""
Conversion of EER (Energy Efficiency Ratio) to SEER2
for Packaged Air Conditioner and Heat Pump.
The same conversion can be used for SEER to SEER2.

See https://www.marathonhvac.com/seer-to-seer2
"""
