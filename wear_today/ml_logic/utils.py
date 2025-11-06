from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
import pandas as pd
import requests


def get_coords_from_location_name(location: str = "Berlin, Germany", timeout: int = 5):
    """
    Translates place name like "Berlin, Germany" into (latitude, longitude)
    """
    geolocator = Nominatim(user_agent="my_geocoder")
    location = geolocator.geocode(location, timeout=timeout)
    if location is None:
        print("Location could not be resolved. Using 'Berlin, Germany' instead.")
        location = geolocator.geocode("Berlin, Germany")
    return (location.latitude, location.longitude)


def get_coords_from_location_name_dummy(location: str = "Berlin, Germany"):
    """
    Returns the coordinates (lat, lon) and timezone of a known city among a hard coded
    dictionnary
    If the city is not found, returns None and prints a warning message.
    """
    # Internal dictionary of coordinates
    city_coords = {
        "Berlin, Germany": {"lat": 52.5200, "lon": 13.4050, "timezone": "Europe/Berlin"},
        "Marseille, France": {"lat": 43.2965, "lon": 5.3698, "timezone": "Europe/Paris"},
        "Porto, Portugal": {"lat": 41.1579, "lon": -8.6291, "timezone": "Europe/Lisbon"},
        "London, United Kingdom": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"}
    }

    coords = city_coords.get(location)
    if coords:
        return coords["lat"], coords["lon"], coords['timezone']
    else:
        print(f"⚠️ City '{location}' not found.")
        print(f"Available cities: {', '.join(city_coords.keys())}")
        return None


def get_timezone_from_coords(latitude, longitude):
    """
    Translates (latitude, longitude) into timezone name like "Europe/Berlin"
    """

    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=latitude, lng=longitude)
    return tz_name


def describe_by_threshold(value, thresholds):
    """
    Return the first matching description based on threshold values.
    """

    for limit, desc in thresholds:
        if value < limit:
            return desc
    return thresholds[-1][1]  # default (last) description


def describe_weather(weather_info: dict):
    """
    Generate a natural language description of the weather conditions.
    """

    time = weather_info["time"].hour
    hum = weather_info["humidity"]
    humid = describe_by_threshold(
        hum,
        [
            (30, "extremely dry"),
            (40, "rather dry"),
            (60, "neither dry nor humid"),
            (70, "humid"),
            (85, "pretty humid"),
            (float("inf"), "moist"),
        ],
    )

    temp = weather_info["temperature"]
    warmth = describe_by_threshold(
        temp,
        [
            (-10, "insufferably cold, way below freezing"),
            (0, "below freezing"),
            (10, "rather cold and uncomfortable"),
            (15, "cool and fresh"),
            (18, "fresh and slightly uncomfortable"),
            (22, "mild and pleasant"),
            (26, "warm"),
            (30, "very warm, almost hot"),
            (float("inf"), "hot and unpleasant"),
        ],
    )

    wind = weather_info["wind"]
    windy = describe_by_threshold(
        wind,
        [
            (0, "perfectly windstill"),
            (5, "very calm"),
            (10, "slightly breezy"),
            (20, "breezy and gusty"),
            (30, "stormy and very gusty"),
            (float("inf"), "the time of a storm"),
        ],
    )

    rain = weather_info["rain"]
    raining = describe_by_threshold(
        rain,
        [
            (0, "not raining, the air is clear"),
            (1, "slightly drizzling"),
            (5, "raining"),
            (10, "raining heavily"),
            (float("inf"), "raining cats and dogs"),
        ],
    )

    sn = f"At around {time} o'clock the air is {warmth} at {temp} °C, it is {windy} with a windspeed of {wind} km/h. "
    sn += f"The air feels {humid} with {hum} % humidity, it is {raining}."

    return sn


def link_is_working(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        )}
    try:
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        return response.status_code < 400
    except requests.RequestException:
        return False
