import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from pathlib import Path
from wear_today.ml_logic.params import *
from wear_today.ml_logic.utils import *


class ClothingRecommender:
    """
    Clothing Recommender Class
    """
    def __init__(
        self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.classifier = None
        self.clothing_df = None
        self.embedder = None

        current_file = Path(__file__).resolve()
        root_dir = current_file.parent.parent.parent
        self.cache_dir = os.path.join(root_dir, MODELS_DIRECTORY)
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self):
        """
        Load clothing data
        """
        current_file = Path(__file__).resolve()
        encoding_data_path = os.path.join(current_file.parent.parent, DIR_PREPROC_CLOTHES)

        df_accessories = pd.read_csv(
            os.path.join(encoding_data_path, "accessories.csv")
        )
        df_shoes = pd.read_csv(
            os.path.join(encoding_data_path, "shoes.csv")
        )
        df_tops = pd.read_csv(os.path.join(encoding_data_path, "top.csv"))
        df_bottoms = pd.read_csv(
            os.path.join(encoding_data_path, "bottom.csv")
        )

        self.df_clothes = pd.concat((df_accessories, df_shoes, df_tops, df_bottoms))

        print(f"✅ Loaded {len(self.df_clothes)} clothing items")
        '''if embeddings file exists load embeddings,
        otherwise save embeddings'''

        self.clothes_embeddings = {}
        for clothes_cat in self.df_clothes['category_type'].unique():
            file_path = os.path.join(encoding_data_path, f"embeddings_{clothes_cat}.npy")
            if os.path.exists(file_path):
                print(f"✅ Loading wardrobe embeddings for {clothes_cat}!")
                self.clothes_embeddings[clothes_cat] = np.load(file_path)
            else:
                print(f"✅ Embedding wardrobe for {clothes_cat}!")
                mask = list(self.df_clothes['category_type'] == clothes_cat)
                self.clothes_embeddings[clothes_cat] = self.embedder.encode(
                    self.df_clothes.iloc[mask]['text_complex']
                    )
                np.save(file_path, self.clothes_embeddings[clothes_cat])

        return self

    def initialize_clothesmodel(self):
        """
        Initialize the sentence embedding model with caching
        """
        if self.embedder is not None:
            return self

        model_path = os.path.join(self.cache_dir, "all-MiniLM-L6-v2")
        print(model_path)
        if os.path.exists(model_path):
            print("📂 Loading cached model...")
            try:
                self.embedder = SentenceTransformer(model_path)
                print("✅ Cached model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Error loading cached model, downloading fresh: {e}")
                self._download_and_cache_model()
        else:
            print("⬇️  Downloading and caching model...")
            self._download_and_cache_model()

        return self

    def _download_and_cache_model(self):
        """
        Download and cache the model
        """
        model_path = os.path.join(self.cache_dir, "all-MiniLM-L6-v2")

        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Save to cache
            print("💾 Saving model to cache...")
            self.embedder.save(model_path)
            print(f"✅ Model cached at: {model_path}")

        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise

    def temperature_to_label(self, temperature):
        """
        Convert numeric temperature to label
        """
        temperature_mapping = {
            (5, 10): "5-10°C cold weather",
            (10, 15): "10-15°C cool weather",
            (15, 20): "15-20°C mild weather",
            (20, 25): "20-25°C warm weather",
            (25, 30): "25-30°C hot weather"
        }

        try:
            temp = float(temperature)
            for (low, high), label in temperature_mapping.items():
                if low <= temp <= high:
                    return label
            if temp < 5:
                return "5-10°C cold weather"
            elif temp > 30:
                return "25-30°C hot weather"
        except ValueError:
            raise ValueError(f"Invalid temperature: {temperature}. Please enter 5-30.")

    def call_embedder(self, input):
        """
        Calls a sentence embedding model and returns scores
        """
        # Single batch classification (fast!)
        print("⚡ Classifying sampled items in one batch...")
        # clothing_emb = self.embedder.encode(input["clothes"])
        weather_emb = self.embedder.encode(input['weather'])
        scores = cosine_similarity([weather_emb], input['clothes_emb'])[0]
        return scores

    def recommend(self, df_weather, top_k=5, sample_size=200):
        """
        Recommend clothing based on weather data
        """
        if self.df_clothes is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.initialize_clothesmodel()

        temp_range = [min(df_weather["temperature"]), max(df_weather["temperature"])]
        humid_range = [min(df_weather["wind"]), max(df_weather["wind"])]
        wind_range = [min(df_weather["humidity"]), max(df_weather["humidity"])]
        rain_range = [min(df_weather["rain"]), max(df_weather["rain"])]
        print(f"🔍 Finding best clothing for {temp_range[0]}°C - {temp_range[1]}°C,")
        print(f"    wind speed of {wind_range[0]}m/s - {wind_range[1]}m/s,")
        print(f"    humidity between {humid_range[0]}% and {humid_range[1]}%,")
        print(f"    and rainfall of max. {rain_range[1]}mm.")
        weather_sentence = describe_weather(
            df_weather.iloc[0]
        )  # TODO: we currently have a sentence for each of the 12 hours of forecast, but only feed 1 to the model

        # give recommendations for top, bottom, shoes, and accessories
        recommendations = []
        for clo_cat in self.df_clothes['category_type'].unique():
            mask = self.df_clothes["category_type"] == clo_cat
            wardrobe = self.df_clothes[mask]
            # RANDOM SAMPLE - This is the key speed improvement!
            if len(wardrobe) > sample_size:
                sample_idx = np.random.choice(wardrobe.index, size=sample_size, replace=False)
                print(
                    f"🎯 Sampling {sample_size} random items from {len(wardrobe)} total items"
                )
            else:
                sample_idx = np.random.choice(wardrobe.index, size=len(wardrobe), replace=False)
                print(f"🎯 Using all {len(wardrobe)} items")

            # select corresponding rows from embeddings
            sample_df = wardrobe.iloc[sample_idx]
            clothes_emb_sample = self.clothes_embeddings[clo_cat][sample_idx,:]

            # Single batch classification (fast!)
            input = {
                "weather": weather_sentence,  # TODO: we currently have a sentence for each of the 12 hours of forecast, but only feed 1 to the model
                "clothes_emb": clothes_emb_sample,
            }

            scores = self.call_embedder(input)

            # Handle results
            high_score_ix = np.argsort(scores)[::-1][:top_k]
            # Build results
            for i in range(top_k):
                item = sample_df.iloc[high_score_ix[i]]
                recommendations.append(
                    {
                        "category": clo_cat,
                        "rank": i + 1,
                        "product_name": item["product_name"],
                        "gender": item["gender"],
                        "details": item["details"],
                        "images": item["product_images"],
                        "confidence": round(scores[high_score_ix[i]], 3),
                        "temperature_range": temp_range,
                        "humidity_range": humid_range,
                        "windspeed_range": wind_range,
                        "rainfall_range": rain_range,
                        "sampled_from_total": f"{len(sample_df)}/{len(wardrobe)} items",
                        "links": re.findall(r"\{'(.*?)': ", item['product_images']),
                    }
                )

        return pd.DataFrame(recommendations)


if __name__ == "__main__":
    input = {
        "time": [datetime.now() + timedelta(hours=i) for i in range(6)],
        "humidity": [60, 50, 50, 50, 60, 70],
        "temperature": [11, 11, 12, 12, 13, 13],
        "wind": [15, 16, 18, 20, 22, 30],
        "rain": [0, 0, 0, 0, 1.0, 1.5],
    }
    df = pd.DataFrame(input)
    recommender = ClothingRecommender()
    recommender.initialize_clothesmodel()
    recommender.load_data()
    recommendations = recommender.recommend(df)
    print(recommendations)
