# This is a lightweight implementation of the ML model in fastapi

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List

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
    ocean_proximity: str

    # Adding the dataframe completion:
    def to_dataframe(self) -> pd.DataFrame:
            data = self.dict()
            # 1. Calculate the bedroom_ratio
            data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
            
            # 2. Create The DataFrame
            df = pd.DataFrame([data])
            
            # 3. Map ocean_proximity to numerical values
            df["ocean_proximity"] = df["ocean_proximity"].map({"1H OCEAN": 0, "INLAND": 1, "NEAR OCEAN": 2, "NEAR BAY": 3})
            
            return df
    
# Loading the single model. 
with open('.\\models\\single_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Creating the decorators
@app.post('/')
# TODO: Fix the post request since the dictionary is not working. 
async def predict(features: PredictionFeatures):
    df = features.to_dataframe()
    prediction = model.predict(df)
    
    return {"prediction": prediction.tolist()}