# TODO: Import your package, replace this by explicit imports of what you need
# from wear_today.main import predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from wear_today.interface.main import recommend_cloths

app = FastAPI()
# TODO: Define how to load the model
# app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {"message": "Hi, The API is running!"}


# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def predict(city: str):

    # TODO: feed it to your model.predict, and return the output

    # X_pred = dict(
    #         day=str(datetime.today().date()),
    #         city=city)
    # index=[0])
    # model = load_model() #function to be confirmed
    # assert model is not None
    weather_prediction, recommended_clothing = recommend_cloths_for_today(
        city=city, today=datetime.today().date()
    )
    # y_pred = X_pred['day']
    # app.state.model.predict(X_pred)

    return f"Today in {city} for {weather_prediction}, my recommendation for your clothing is: {recommended_clothing}"
