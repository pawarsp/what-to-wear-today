import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from wear_today.ml_logic.params import *
from wear_today.ml_logic.data import *
from wear_today.ml_logic.recommender_model import ClothingRecommender
from wear_today.ml_logic.weather_model import WeatherPredictor


def recommend_clothes_for_today(city="Berlin", today=datetime.now()):
    """
    Get clothing recommendation based on city weather for today.
    """
    city_country = {
        "london": "London, United Kingdom",
        "berlin": "Berlin, Germany",
        "porto": "Porto, Portugal",
        "marseille": "Marseille, France",
    }
    city_name = city_country[city.lower()]

    # receive 12 hour weather prediction
    # obtain clothing recommendation
    print(f"Gathering clothing recommendation for {city_name}")

    weather_predictor = WeatherPredictor(location=city_name)
    clothing_recommender = ClothingRecommender()
    clothing_recommender.initialize_clothesmodel()
    clothing_recommender.load_data()
    weather_prediction = weather_predictor.predict()
    recommended_clothing = clothing_recommender.recommend(weather_prediction, top_k=1, sample_size=10)

    weather_prediction_dict = weather_prediction.to_dict(orient="list")
    recommended_clothing_dict = recommended_clothing.to_dict(orient="list")

    times = weather_prediction_dict["time"]

    weather_clothes_dict = {
        "time": [f"{t.hour:02d}:{t.minute:02d}" for t in times],
        "temperature": list(np.round(weather_prediction_dict["temperature"], 2)),
        "rain": list(np.round(weather_prediction_dict["rain"], 2)),
        "humidity": list(np.round(weather_prediction_dict["humidity"], 2)),
        "wind": list(np.round(weather_prediction_dict["wind"], 2)),
        "temperature_min": np.round(min(weather_prediction_dict["temperature"]), 2),
        "temperature_max": np.round(max(weather_prediction_dict["temperature"]), 2),
        "recommended_clothes": {
            "accessories": recommended_clothing_dict["product_name"][0],
            "shoes": recommended_clothing_dict["product_name"][1],
            "top": recommended_clothing_dict["product_name"][2],
            "bottom": recommended_clothing_dict["product_name"][3],
            "accessories_link": recommended_clothing_dict["links"][0],
            "shoes_link": recommended_clothing_dict["links"][1],
            "top_link": recommended_clothing_dict["links"][2],
            "bottom_link": recommended_clothing_dict["links"][3]
        }
    }
    return weather_clothes_dict


if __name__ == "__main__":
    weather_clothes_dict = recommend_clothes_for_today('london')
