import json
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI(
    title="Pearls AQI Predictor API",
    description=(
        "Prediction API for Lahore AQI forecasts "
        "at +24 h, +48 h and +72 h."
    ),
    version="1.0.0"
)


SNAPSHOT_FILE = "latest_prediction_snapshot.json"


def load_predictions():

    if not os.path.exists(SNAPSHOT_FILE):
        raise FileNotFoundError(
            "Prediction snapshot is unavailable."
        )

    with open(
        SNAPSHOT_FILE,
        "r"
    ) as file:

        data = json.load(file)

    return data


@app.get("/")
def root():

    return {
        "service": "Pearls AQI Predictor API",
        "city": "Lahore",
        "status": "running",
        "endpoints": [
            "/health",
            "/predict",
            "/docs"
        ]
    }


@app.get("/health")
def health():

    if os.path.exists(SNAPSHOT_FILE):

        return {
            "status": "healthy",
            "prediction_data": "available"
        }

    return JSONResponse(
        status_code=503,
        content={
            "status": "degraded",
            "prediction_data": "unavailable"
        }
    )


@app.get("/predict")
def predict():

    try:

        data = load_predictions()

        return {
            "city": data.get(
                "city",
                "Lahore"
            ),

            "last_updated":
                data.get(
                    "source_timestamp"
                ),

            "current_aqi":
                data.get(
                    "current_aqi"
                ),

            "forecast": {

                "24_hours":
                    data.get(
                        "aqi_24h"
                    ),

                "48_hours":
                    data.get(
                        "aqi_48h"
                    ),

                "72_hours":
                    data.get(
                        "aqi_72h"
                    )
            },

            "models":
                data.get(
                    "models"
                ),

            "data_mode":
                "latest successful prediction snapshot"
        }

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(error)
            }
        )
    