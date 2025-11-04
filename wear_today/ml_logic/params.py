import os
import numpy as np

##### VARIABLES #####
METEO_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "rain",
    "wind_speed_10m",
    "cloud_cover",
    "pressure_msl",
]
RAW_DATA_PATH = os.path.join(
    os.path.expanduser("~"), "code", "pawarsp", "what-to-wear-today", "raw_data"
)
DIR_PREPROC_CLOTHES = "sample_data"

####WEATHER MODEL PARAMS#########
MODEL_DIR = "models"

TARGET = [
    'temperature_2m',
    'relative_humidity_2m',
    'rain',
    'wind_speed_10m'
]
N_TARGETS = len(TARGET)
INPUT_LENGTH = 336 #Nb of hours used to predict next hours
OUTPUT_LENGTH = 12 #Nb of hours to forecast
