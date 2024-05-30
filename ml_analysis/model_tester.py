import pickle
import pandas as pd

# Load the full pipeline from the pickle file
with open('.\\models\\single_model.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

print("Full pipeline model loaded from full_pipeline_model.pkl")

# Example new data (replace with your actual new data)
X_new = pd.DataFrame({
    'longitude': [-122.23, -122.22],
    'latitude': [37.88, 37.86],
    'housing_median_age': [41.0, 21.0],
    'total_rooms': [880.0, 7099.0],
    'total_bedrooms': [129.0, 1106.0],
    'population': [322.0, 2401.0],
    'households': [126.0, 1138.0],
    'median_income': [8.3252, 8.3014],
    'ocean_proximity': ['NEAR BAY', 'NEAR BAY']
})

X_new['bedroom_ratio'] = X_new['total_bedrooms'] / X_new['total_rooms']
X_new["ocean_proximity"] = X_new["ocean_proximity"].map({"1H OCEAN": 0, "INLAND" : 1, "NEAR OCEAN" : 2, "NEAR BAY" : 3})
 
# Make predictions using the loaded pipeline
predictions = loaded_pipeline.predict(X_new)

print("Predictions on new data:", predictions)