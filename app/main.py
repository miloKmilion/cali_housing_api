# This is a lightweight implementation of the ML model in fastapi

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict

app = FastAPI()

class PredictionFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: int
    bedroom_ratio: float


with open('.\\models\\single_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Creating the decorators
@app.post('/')
# TODO: Fix the post request since the dictionary is not working. 
async def scoring_endpoint(item: PredictionFeatures):
    df = pd.DataFrame([item.dict().values()], item.dict().keys()).T
    # df = pd.DataFrame([item.model_dump()])
    # Ensure df is a DataFrame
    
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    print(df)
    
    prediction = model.predict(df)

    # Return the prediction
    return {"prediction": prediction}