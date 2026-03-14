

# 👗 What To Wear Today

**ML-powered All Weather Clothing Recommender**


An end-to-end machine learning pipeline that forecasts the next 12 hours of weather, then recommends what to wear based on those conditions.

---

## 🔍 How It Works

```
Open-Meteo API  →  Clean & cache weather data
        ↓
LSTM model  →  12-hour weather forecast
        ↓
Weather  →  natural language description
        ↓
Sentence embeddings  →  cosine similarity vs. clothing catalogue
        ↓
Top 5 recommendations per clothing category
```

The system runs in **two ML stages**:

**Stage 1 - Weather Forecasting**
An LSTM neural network trained on 14 days of hourly historical data predicts temperature, humidity, rain, and wind speed for the next 12 hours.

**Stage 2 - Clothing Recommendation**
The forecast is converted into a natural language sentence, then matched against a clothing catalogue using semantic cosine similarity via the `all-MiniLM-L6-v2` sentence transformer.

---

## 🌍 Supported Cities

| City | Country | Timezone |
|------|---------|----------|
| Berlin | Germany | Europe/Berlin |
| Marseille | France | Europe/Paris |
| Porto | Portugal | Europe/Lisbon |
| London | United Kingdom | Europe/London |


## 🧠 Model Details

### Weather Predictor - LSTM

| Parameter | Value |
|-----------|-------|
| Input length | 336 hours (14 days) |
| Output length | 12 hours |
| Input features | temperature, humidity, dew point, apparent temperature, rain, wind speed, cloud cover, pressure |
| Target features | temperature, humidity, rain, wind speed |
| Date encoding | Cyclical sin/cos (hour, day of week, day of year) |
| Data source | Open-Meteo Archive API (cached locally as CSV) |
| Missing data | Filled via linear interpolation |

### Clothing Recommender - Sentence Transformers

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` |
| Similarity metric | Cosine similarity |
| Categories | Tops, Bottoms, Shoes, Accessories |
| Recommendations | Top 5 per category |
| Sample size | Up to 200 items per category (for speed) |
| Embeddings | Pre-computed and cached as `.npy` files |
| Image validation | Product image links verified before returning |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `tensorflow` / `keras` | LSTM weather prediction model |
| `sentence-transformers` | Clothing recommendation via semantic embeddings |
| `openmeteo-requests` | Weather data API client |
| `fastapi` / `uvicorn` | REST API server |
| `scikit-learn` | Data scaling & cosine similarity |
| `pandas` / `numpy` | Data processing and manipulation |
| `geopy` / `timezonefinder` | Location and timezone resolution |
| `joblib` | Scaler serialisation |
| `streamlit` | Frontend UI |


``
# Getting started

Package name: wear_today

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
    


