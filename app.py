
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

# set the output as pandas
set_config(transform_output='pandas')

# initialize dagshub
import dagshub
import mlflow.client

dagshub.init(repo_owner='manikantmnnit', repo_name='swiggy-delivery-time-prediction', mlflow=True)

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/manikantmnnit/swiggy-delivery-time-prediction.mlflow")


class Data(BaseModel):
    age: float = 26.0
    ratings: float = 4.2
    weather: str = "Cloudy"
    traffic: str = "Medium"
    vehicle_condition: int = 1
    type_of_order: str = "Snack"
    type_of_vehicle: str = "motorcycle"
    multiple_deliveries: float = 1.0
    festival: str = "No"
    city_type: str = "Metropolitian"
    is_weekend: int = 0
    pickup_time_minutes: float = 15.0
    order_time_of_day: str = "18:50:00"
    distance: float = 2.5
    distance_type: str = "short"


    
    
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


def load_transformer(transformer_path):
    transformer = joblib.load(transformer_path)
    return transformer



# columns to preprocess in data
num_cols = ["age",
            "ratings",
            "pickup_time_minutes",
            "distance"]

nominal_cat_cols = ['weather',
                    'type_of_order',
                    'type_of_vehicle',
                    "festival",
                    "city_type",
                    "is_weekend",
                    "order_time_of_day"]

ordinal_cat_cols = ["traffic","distance_type"]

#mlflow client
client = MlflowClient()

# load the model info to get the model name
model_name = load_model_information("run_information.json")['model_name']

# stage of the model
stage = "Staging"

# get the latest model version
# latest_model_ver = client.get_latest_versions(name=model_name,stages=[stage])
# print(f"Latest model in production is version {latest_model_ver[0].version}")

# load model path
model_path = f"models:/{model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# load the preprocessor
preprocessor_path = "models/preprocessor.joblib"
preprocessor = load_transformer(preprocessor_path)

# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return "Welcome to the Swiggy Food Delivery Time Prediction App"

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data): 
    pred_data = pd.DataFrame({
        'age': data.age,
        'ratings': data.ratings,
        'weather': data.weather,
        'traffic': data.traffic,
        'vehicle_condition': data.vehicle_condition,
        'type_of_order': data.type_of_order,
        'type_of_vehicle': data.type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'festival': data.festival,
        'city_type': data.city_type,
        'is_weekend': data.is_weekend,
        'pickup_time_minutes': data.pickup_time_minutes,
        'order_time_of_day': data.order_time_of_day,
        'distance': data.distance,
        'distance_type': data.distance_type
    }, index=[0]) 
    
  
    # # clean the raw input data
    # cleaned_data = perform_data_cleaning(pred_data)
    # get the predictions
    predictions = model_pipe.predict(pred_data)[0]

    return predictions
   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app")