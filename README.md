# what-to-wear-today
What-to-wear-today is the app that ends your morning struggle. It estimates the weather for the next 12 hours for your current location and overrides your wardrobe malfunction. No more choosing! Get recommended 4 items for your head, feet, legs, and chest to dress for any occasion. Whether hail, sleet, snow, or sunshine: You are prepared!

ML-powered All Weather Clothing Recommender

Package name: wear_today

# Getting started

## Expected folder tree
    .
    ├── api
    ├── models
    ├── notebook
    ├── raw_data
    ├── scripts
    ├── tests
    └── wear_today
  
## Installation

Use terminal and install the `wear_today` package using pip. Make sure your `what_to_wear_today` pyenv is active.

    cd what-to-wear-today
    
    pip install -e .

## Check if installation was successful
Use terminal

    pip list | grep wear_today


## Docker Image

### Build
    cd what-to-wear-today
    docker build -t api .   
### Run
    cd what-to-wear-today
    docker run -p 8000:8000 -e PORT=8000 api
    


