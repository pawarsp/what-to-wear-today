import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from pathlib import Path
from retry_requests import retry
from .utils import get_coords_from_location_name, get_timezone_from_coords
from .params import *


def get_meteo_data_with_cache(
    start_date: str = "2023-01-01",
    end_date: str = "2025-10-15",
    location: str = "Berlin, Germany",
):
    """
    Retrieve data through the Open-Meteo API (), or from local file, if the file exists
    Stores data at raw_data/weatherdata_startdate_enddate_location.csv if retrieved from API for future use
    Returns a pandas.DataFrame
    """

    latitude, longitude = get_coords_from_location_name(location)
    timezone = get_timezone_from_coords(latitude, longitude)
    filepath = Path(RAW_DATA_PATH).joinpath(
        f'{location.replace(", ", "_").lower()}_{start_date.replace("-", "")}_{end_date.replace("-", "")}.csv'
    )

    if filepath.is_file():
        print("Load meteo data from local CSV.")
        df = pd.read_csv(filepath, header="infer")
        if ((len(METEO_VARIABLES) + 1) != df.shape[1]) or not all(
            [x in df.columns for x in METEO_VARIABLES]
        ):
            print(
                "Variables from local file are different from desired variables. Deleting local data."
            )
            os.remove(filepath)
            return get_meteo_data_with_cache(start_date, end_date, location)
    else:
        print("Load meteo data from Open-Meteo server.")

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": METEO_VARIABLES,
            "timezone": timezone,
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        for idx, var in enumerate(METEO_VARIABLES):
            data[var] = hourly.Variables(idx).ValuesAsNumpy()
        df = pd.DataFrame(data)

        # save data to csv
        if df.shape[0] > 1:
            df.to_csv(filepath, index=False)
    return df


def get_input_weather_data():
    """
    Function to get 2 week time series weather data required for weather prediction
    """
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - filling missing data with interpolated data
    """

    # Interpolate missing data
    df = fill_missing_timestamps(df)

    print("✅ data cleaned")

    return df


def fill_missing_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if some rows are missing; fill them with linear interpolation and return
    the full dataframe.
    """
    # Set date to index to build a date range with full datetime
    df.set_index("date", inplace=True, drop=True)
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")

    # Detect the missing timestamps
    missing_timestamps = full_index.difference(df.index)

    # Define new indices for our df with the whol list of indices and
    # interpolate missing values
    df = df.reindex(full_index)
    df = df.interpolate(method="linear")
    df.index.name = "date"
    if len(missing_timestamps) != 0:
        print(f"{len(missing_timestamps)} missing timestamp(s) interpolated")
    # Return the cleaned dataframe and put back 'date' as a column
    return df.reset_index().rename(columns={"index": "date"})
