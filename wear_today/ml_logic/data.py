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
