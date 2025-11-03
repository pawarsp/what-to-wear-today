import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from colorama import Fore, Style

from keras.models import load_model
from keras import losses
import joblib

from wear_today.ml_logic.params import *
from wear_today.ml_logic.data import *

from pathlib import Path


"""Weather Predictor Class"""


class WeatherPredictor:
    def __init__(self, location='Berlin, Germany', today=datetime.now()):
        self.location = location
        self.today = today


    def describe_now(self):
        date_str = self.today.strftime("%A %d %B %Y")
        time_str = self.today.strftime("%H:%M:%S")
        location = self.location
        print(
            f"Today is {date_str}, the time is {time_str}.\
                Let's forecast the weather data for the next 12 hours in {location}.")

    def load_and_clean_last_data(self, input_hours=INPUT_LENGTH) -> pd.DataFrame:
        '''
        Load latest data for input_length/24 days before today
        '''
        #Convert nb of hours from input_hours into nb of inopout days to load
        input_length_days = input_hours // 24   + 1* (input_hours % 24 >0)

        start_date = str(self.today.date() - timedelta(days = input_length_days))
        end_date = str(self.today.date())
        location = self.location

        #Load Data from Open Meteo API and clean
        df_weather = get_meteo_data_with_cache(start_date, end_date, location)
        df_weather_cleaned = clean_data(df_weather)

        #Filter Data to only keep data up to the last hour
        latitude, longitude = get_coords_from_location_name(location= "Berlin, Germany")
        input_timezone = get_timezone_from_coords(latitude, longitude)

        #Check if date is UTC or timezone formatted
        if df_weather_cleaned["date"].dt.tz is None:
            # tz-naive → localize it
            df_weather_cleaned["date"] = df_weather_cleaned["date"].dt.tz_localize("UTC")
        else:
            # tz-aware → convert it to the target timezone if needed
            df_weather_cleaned["date"] = df_weather_cleaned["date"].dt.tz_convert(input_timezone)

        df_weather_cleaned = df_weather_cleaned[df_weather_cleaned.date <= pd.Timestamp.now(tz=input_timezone)]
        #####Note : using pd.Timestamp.now(tz=input_timezone) is not really good since pd.Timestamp.now()
        #will be different than today's input...we could check that later and maybe place UTC in the input
        #and replace pd.Timestamp.now(tz=input_timezone) by self.today


        #Filter Data to only keep last #input_hours hours of data
        df_weather_filtered = df_weather_cleaned.iloc[-input_hours:, :]

        return df_weather_filtered


    def load_model(self):
        '''Get the latest model of the weather predictor.
        Return None if no model found
        '''

        # Get the latest model version name by the timestamp
        #import ipdb; ipdb.set_trace()

        current_file = Path(__file__).resolve()
        root_dir = current_file.parent.parent.parent
        model_dir= os.path.join(root_dir, MODELS_DIRECTORY)

        #model_directory = MODELS_DIRECTORY
        model_paths = glob.glob(os.path.join(model_dir, "weather_predictor*"))
        print("Searching models in:", model_dir)
        print("Found models:", model_paths)

        if not model_paths:
            return None

        most_recent_model_path = sorted(model_paths)[-1]

        print(Fore.BLUE + "\nLoad latest weather forecast model from Docker image..." + Style.RESET_ALL)

        latest_model = load_model(most_recent_model_path, compile=True,
                   custom_objects={'mse': losses.MeanSquaredError()})

        print("✅ Model loaded from Docker image")

        return latest_model

    def preprocessing(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Transform datetime date into cyclical features for Hour, day of week and day of year
        Drop date column
        Return preprocessed dataframe
        '''
        df = raw_data.copy()
        weather_date = pd.to_datetime(df["date"])

        df["day_of_year_sin"] = np.sin(2 * np.pi * weather_date.dt.dayofyear / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * weather_date.dt.dayofyear / 365)
        df["day_of_week_sin"] = np.sin(2 * np.pi * weather_date.dt.weekday / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * weather_date.dt.weekday / 7)
        df["hour_sin"] = np.sin(2 * np.pi * weather_date.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * weather_date.dt.hour / 24)

        # Drop 'date' column : we do not need it anymore
        df = df.drop(columns=["date"])

        # Add an extra dimension at axis 0 to create a batch dimension,
        # so that the input shape becomes (batch_size, sequence_length, features)
        df = np.expand_dims(df, axis=0)

        return df

    def predict(self):
        """
        Predict weather for the next hours using the latest trained LSTM model.
        Returns predictions in original scale (unscaled).
        """
        # Load model
        model = self.load_model()
        if model is None:
            raise ValueError("No model found. Please train a model first.")

        # Load scaler
        scaler_y = self.load_scaler()
        if scaler_y is None:
            raise ValueError("No scaler found. Please train a model and save the scaler first.")

        # Load and preprocess latest data
        df_clean = self.load_and_clean_last_data()
        df_processed = self.preprocessing(df_clean)

        # Predict
        y_pred_scaled = model.predict(df_processed)
        y_pred_scaled = np.squeeze(y_pred_scaled)  # remove batch dimension

        # Unscale predictions
        y_pred_original = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, N_TARGETS)
        ).reshape(y_pred_scaled.shape)

        #Format the output
        df_pred = pd.DataFrame(
            y_pred_original,
            columns=TARGET
        )

        #Add the date dimension --> to do

        print(Fore.GREEN + f"✅ Prediction completed with shape {y_pred_original.shape}" + Style.RESET_ALL)

        return df_pred

    def save_model(self):
        pass

    def load_scaler(self):
        '''Get the latest target scaler used in the latest trained model.
        Necessary for unscaling prediction from weather forecast model.
        Return None if no scaler found
        '''

        # Get the latest scaler version name by the timestamp
        scaler_directory = MODELS_DIRECTORY
        scaler_paths = glob.glob(os.path.join(scaler_directory, "weather_y_scaler*"))

        if not scaler_paths:
            return None

        most_recent_scaler_path = sorted(scaler_paths)[-1]

        print(Fore.BLUE + "\nLoad latest target scaler from Docker image..." + Style.RESET_ALL)

        latest_scaler = joblib.load(most_recent_scaler_path)

        print("✅ Scaler loaded from Docker image")

        return latest_scaler


    def save_scaler(self):
        pass




if __name__ == "__main__":
    predictor = WeatherPredictor() #ensuite tester en changeant date
    predictions = predictor.predict()
    print(predictions)
