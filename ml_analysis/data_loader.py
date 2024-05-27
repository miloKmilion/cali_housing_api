import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DataLoader:
    folder_name: str
    filename: str
    final_path: Optional[str] = field(init=False, repr=False)
    

    def __post_init__(self):
        self.final_path = os.path.join(self.folder_name, self.filename)
    
    def load_data(self) -> pd.DataFrame:
        dataframe_loaded = None  # Initialize dataframe_loaded

        if self.filename.endswith(".csv"):
            dataframe_loaded = pd.read_csv(self.final_path)
            lowercase = lambda x: str(x).lower()
            dataframe_loaded.rename(lowercase, axis='columns', inplace=True)
            print(f"The original dataframe has this number of columns: {dataframe_loaded.shape[1]} and rows: {dataframe_loaded.shape[0]}")
            
        elif self.filename.endswith(".parquet"):
            dataframe_loaded = pd.read_parquet(self.final_path)
            lowercase = lambda x: str(x).lower()
            dataframe_loaded.rename(lowercase, axis='columns', inplace=True)
            print(f"The original dataframe has this number of columns: {dataframe_loaded.shape[1]} and rows: {dataframe_loaded.shape[0]}")
        
        else:
            raise ValueError(f"Unsupported file format: {self.filename}")
            
        return dataframe_loaded
    
    @staticmethod
    def initial_data_manipulatiuon(dataframe_loaded: pd.DataFrame) -> None:
        print(f"Shape of the dataframe: {dataframe_loaded.shape} \n")
        print(f"Columns in the dataframe: {dataframe_loaded.columns} \n")
        print(dataframe_loaded.isnull().sum())
      
        #Showing missing data with heatmap before droping 
        plt.figure(figsize=(10,6))
        sns.heatmap(dataframe_loaded.isna().transpose());           