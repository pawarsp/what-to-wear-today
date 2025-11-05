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


def recommend_clothes_for_today(city="Berlin, Germany", today=datetime.now()):
    """
    Get clothing recommendation based on city weather for today.
    """

    # receive 12 hour weather prediction
    # obtain clothing recommendation
    print(f"Gathering clothing recommendation for {city}")
    weather_predictor = WeatherPredictor(location = city, today = today)
    clothing_recommender = ClothingRecommender()
    clothing_recommender.load_data()
    weather_prediction = weather_predictor.predict()
    recommended_clothing = clothing_recommender.recommend(weather_prediction, top_k=1, sample_size=10)

    return weather_prediction, recommended_clothing


if __name__ == "__main__":
    recommend_clothes_for_today()
