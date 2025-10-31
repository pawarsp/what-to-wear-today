# what-to-wear-today

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
    


