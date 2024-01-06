# sensor_app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os

from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file, load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.constant.application import APP_HOST, APP_PORT
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from fastapi.responses import Response

env_file_path = os.path.join(os.getcwd(), "env.yaml")


def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL', None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase the maximum allowed request size
app = FastAPI(max_request_size=100 * 1024 * 1024)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return JSONResponse(content={"message": "Training pipeline is already running."})
        train_pipeline.run_pipeline()
        return JSONResponse(content={"message": "Training successful !!"})
    except Exception as e:
        return JSONResponse(content={"error": f"Error Occurred! {e}"})


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            raise HTTPException(status_code=500, detail="Model is not available")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        # Make predictions
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Convert DataFrame to CSV string
        predictions_csv = df.to_csv(index=False)

        # Return CSV file as a response
        return Response(
            content=predictions_csv,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment;filename=predictions.csv"}
        )

    except HTTPException as http_exception:
        raise http_exception  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Occurred! {e}")


def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__ == "__main__":
    # main()
    # set_env_variable(env_file_path)
    import uvicorn

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
