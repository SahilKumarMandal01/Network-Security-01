import os
import sys

import certifi
import pandas as pd
import pymongo

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.pipeline.training_pipeline import TrainingPipeline
from NetworkSecurity.utils.main_utils.utils import load_object
from NetworkSecurity.utils.ml_utils.model.estimator import NetworkModel
from NetworkSecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# -------------------------------------------------------------------
# 🔹 Environment & MongoDB Setup
# -------------------------------------------------------------------
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise ValueError("❌ MONGODB_URL_KEY is not set in the environment variables.")

ca = certifi.where()
client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# -------------------------------------------------------------------
# 🔹 FastAPI App Setup
# -------------------------------------------------------------------
app = FastAPI(title="Network Security API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production ⚠️)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates (for rendering prediction results in HTML)
templates = Jinja2Templates(directory="./templates")

# -------------------------------------------------------------------
# 🔹 Routes
# -------------------------------------------------------------------

@app.get("/", tags=["authentication"])
async def index():
    """Redirect to Swagger UI"""
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def train_route():
    """Trigger model training pipeline"""
    try:
        logging.info("🚀 Starting training pipeline...")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        logging.info("✅ Training completed successfully.")
        return Response(content="Training is successful", media_type="text/plain")
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """Upload CSV, make predictions, and return results as HTML table"""
    try:
        logging.info("📂 Reading uploaded CSV file...")
        df = pd.read_csv(file.file)

        # Load preprocessor & model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Generate predictions
        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        # Save output
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # Render results as HTML table
        table_html = df.to_html(classes="table table-striped", index=False)
        logging.info("✅ Prediction completed successfully.")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        raise NetworkSecurityException(e, sys)


# -------------------------------------------------------------------
# 🔹 Main Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
