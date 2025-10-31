import os
import numpy as np

##### VARIABLES #####
METEO_VARIABLES = ["temperature_2m",
                   "relative_humidity_2m",
                   "dew_point_2m",
                   "apparent_temperature",
                   "precipitation",
                   "wind_speed_10m",
                   "cloud_cover",
                   "pressure_msl"]
RAW_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "pawarsp", "what-to-wear-today", "raw_data")
