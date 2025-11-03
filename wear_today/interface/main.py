import os
import numpy as np
import pandas as pd
from transformers import pipeline
from pathlib import Path
from datetime import datetime
from wear_today.ml_logic.params import *
from wear_today.ml_logic.data import *
from wear_today.ml_logic.weather_sentence import weather_sentence
from wear_today.ml_logic.recommender_model import TemperatureRecommender
from wear_today.ml_logic.weather_model import WeatherPredictor


def recommend_cloths_for_today(city="Berlin, Germany", today=datetime.now()):
    # receive 12 hour weather prediction
    # obtain clothing recommendation
    print(f"Gathering clothing recommendation for {city}")
    weather_predictor = WeatherPredictor()
    clothing_recommender = TemperatureRecommender()
    clothing_recommender.load_data()
    weather_prediction = weather_predictor.predict(today)
    input = {
        "time": [datetime.now() + timedelta(hours=i) for i in range(6)],
        "humidity": [60, 50, 50, 50, 60, 70],
        "temperature": [11, 11, 12, 12, 13, 13],
        "wind": [15, 16, 18, 20, 22, 30],
        "rain": [0, 0, 0, 0, 1.0, 1.5],
    }
    df = pd.DataFrame(input)  # replace with weather_prediction
    recommended_clothing = clothing_recommender.recommend(df, top_k=1, sample_size=10)

    return weather_prediction, recommended_clothing


if __name__ == "__main__":
    recommend_cloths()
