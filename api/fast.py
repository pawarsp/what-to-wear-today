# TODO: Import your package, replace this by explicit imports of what you need
# from wear_today.main import predict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from wear_today.interface.main import recommend_clothes_for_today

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

    output_dict = recommend_clothes_for_today(
        city=city, today=datetime.today().date()
    )


    return output_dict
