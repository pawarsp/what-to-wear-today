import os
import pandas as pd

"""Clothing Recommender Class"""


class TemperatureRecommender:
    def __init__(
        self, model_name="facebook/bart-large-mnli", cache_dir="./model_cache"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.classifier = None
        self.clothing_df = None

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self):
        """
        Load clothing data
        """
        df_accessories = pd.read_csv(
            os.path.join(DIR_PREPROC_CLOTHES, "classified_accessories.csv")
        )
        df_shoes = pd.read_csv(
            os.path.join(DIR_PREPROC_CLOTHES, "classified_shoes.csv")
        )
        df_tops = pd.read_csv(os.path.join(DIR_PREPROC_CLOTHES, "classified_top.csv"))
        df_bottoms = pd.read_csv(
            os.path.join(DIR_PREPROC_CLOTHES, "classified_bottom.csv")
        )

        self.df_clothes = pd.concat((df_accessories, df_shoes, df_tops, df_bottoms))

        print(f"✅ Loaded {len(self.df_clothes)} clothing items")
        return self

    def initialize_classifier(self):
        """
        Initialize the zero-shot classifier with caching
        """
        if self.classifier is not None:
            return self

        model_path = os.path.join(self.cache_dir, "bart-large-mnli")

        if os.path.exists(model_path):
            print("📂 Loading cached model...")
            try:
                self.classifier = pipeline(
                    "zero-shot-classification", model=model_path, device=-1
                )
                print("✅ Cached model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Error loading cached model, downloading fresh: {e}")
                self._download_and_cache_model()
        else:
            print("⬇️  Downloading and caching model...")
            self._download_and_cache_model()

        return self

    def _download_and_cache_model(self):
        """Download and cache the model"""
        model_path = os.path.join(self.cache_dir, "bart-large-mnli")

        try:
            self.classifier = pipeline(
                "zero-shot-classification", model=self.model_name, device=-1
            )

            # Save to cache
            print("💾 Saving model to cache...")
            self.classifier.model.save_pretrained(model_path)
            self.classifier.tokenizer.save_pretrained(model_path)
            print(f"✅ Model cached at: {model_path}")

        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise

    def temperature_to_label(self, temperature):
        """
        Convert numeric temperature to label
        """
        try:
            temp = float(temperature)
            for (low, high), label in self.temperature_mapping.items():
                if low <= temp <= high:
                    return label
            if temp < 5:
                return "5-10°C cold weather"
            elif temp > 30:
                return "25-30°C hot weather"
        except ValueError:
            raise ValueError(f"Invalid temperature: {temperature}. Please enter 5-30.")

    def call_zero_shot(self, input):
        """
        calls a zero shot model
        """
        # Single batch classification (fast!)
        print("⚡ Classifying sampled items in one batch...")
        results = self.classifier(
            input["texts"],
            candidate_labels=input["labels"],
            hypothesis_template=input["hypothesis"],
            multi_label=False,
        )
        return results

    def recommend(self, df_weather, top_k=5, sample_size=200):
        """
        Ultra-fast recommendation by sampling only 200 random items
        """
        if self.df_clothes is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.initialize_classifier()

        temp_range = [min(df_weather["temperature"]), max(df_weather["temperature"])]
        humid_range = [min(df_weather["wind"]), max(df_weather["wind"])]
        wind_range = [min(df_weather["humidity"]), max(df_weather["humidity"])]
        rain_range = [min(df_weather["rain"]), max(df_weather["rain"])]
        print(f"🔍 Finding best clothing for {temp_range[0]}°C - {temp_range[1]}°C,")
        print(f"    wind speed of {wind_range[0]}m/s - {wind_range[1]}m/s,")
        print(f"    humidity between {humid_range[0]}% and {humid_range[1]}%,")
        print(f"    and rainfall of max. {rain_range[1]}mm.")
        weather_sentences = describe_weather(
            df_weather.iloc[0]
        )  # TODO: we currently have a sentence for each of the 12 hours of forecast, but only feed 1 to the model

        # give recommendations for top, bottom, shoes, and accessories
        recommendations = []
        for clo_cat in self.df_clothes.category_type.unique():
            mask = self.df_clothes["category_type"] == clo_cat
            wardrobe = self.df_clothes[mask]
            # RANDOM SAMPLE - This is the key speed improvement!
            if len(wardrobe) > sample_size:
                sample_df = wardrobe.sample(sample_size)  # , random_state=42)
                print(
                    f"🎯 Sampling {sample_size} random items from {len(wardrobe)} total items"
                )
            else:
                sample_df = wardrobe
                print(f"🎯 Using all {len(sample_df)} items")

            # Prepare texts from sampled items
            labels = []
            for idx, row in sample_df.iterrows():
                label = " ".join(
                    [
                        "product name:",
                        row["product_name"],
                        "; keywords:",
                        row["text"],
                        "; suitable for",
                        row["weather_label"],
                    ]
                )
                labels.append(label)

            # Single batch classification (fast!)
            input = {
                "texts": weather_sentences,  # TODO: we currently have a sentence for each of the 12 hours of forecast, but only feed 1 to the model
                "labels": labels,
                "hypothesis": "Oh no, I need to go outside for the next 12 hours. What should I wear? Maybe '{}' is a good suggestion to best the weather?",
            }

            results = self.call_zero_shot(input)

            # Handle results
            scores = results["scores"]
            high_scores = scores[:top_k]
            # Build results
            for ix, val in enumerate(high_scores):
                item_ix = labels.index(results["labels"][ix])
                item = sample_df.iloc[item_ix]
                recommendations.append(
                    {
                        "category": clo_cat,
                        "rank": ix + 1,
                        "product_name": item["product_name"],
                        "details": item["details"],
                        "confidence": round(high_scores[ix], 3),
                        "temperature_range": temp_range,
                        "sampled_from_total": f"{len(sample_df)}/{len(wardrobe)} items",
                    }
                )

        return pd.DataFrame(recommendations)
