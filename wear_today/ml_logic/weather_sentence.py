from datetime import datetime, timedelta
import pandas as pd


def describe_by_threshold(value, thresholds):
    """Return the first matching description based on threshold values."""
    for limit, desc in thresholds:
        if value < limit:
            return desc
    return thresholds[-1][1]  # default (last) description


def describe_weather(row):
    time = row["time"].hour
    hum = row["humidity"]
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

    temp = row["temperature"]
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

    wind = row["wind"]
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

    rain = row["rain"]
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
