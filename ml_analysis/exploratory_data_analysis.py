import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class ExploratoryDataAnalysis():
    raw_dataframe: pd.DataFrame
    na_fill_method: str
    target_column: str
    folder_name_to_save: str
    
    def fill_na(self) -> pd.DataFrame:
        if self.na_fill_method == 'drop':
            df_cleaned = self.raw_dataframe.dropna()
        elif self.na_fill_method == 'ffill':
            df_cleaned = self.raw_dataframe.fillna(method='ffill')
        elif self.na_fill_method == 'median':
            df_cleaned = self.raw_dataframe.fillna(self.raw_dataframe.median(numeric_only=True))
        else:
            raise ValueError(f"Unsupported NA fill method: {self.na_fill_method}")
        
        return df_cleaned


    def analyze_data(self, df: pd.DataFrame) -> None:
        print(f"Shape of the dataframe: {df.shape}")
        print(f"Are there duplicates in the dataframe: {df.duplicated().sum()}")
        
        # Get descriptive statistics and round to 2 decimal places
        description = df.describe().round(2)
        print("Description of the dataframe:\n", description)
    
    
    @staticmethod
    def visualize_data(df: pd.DataFrame) -> None:
        # Plot distributions
        plt.figure(figsize=(10, 15))
        for i, col in enumerate(df.columns, 1):
            plt.subplot(5, 2, i)
            sns.histplot(x=df[col], kde=True)
            plt.title(f"Distribution of {col} Data")
            plt.tight_layout()
            plt.xticks(rotation=90)
        plt.show()

        # Plot box plots
        df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(15, 7))
        plt.show()

        # Plot heatmap of correlations
        df_corr = df.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_corr, fmt=".3f", annot=True, cmap="YlGnBu")
        plt.show()
    
    
    def feature_engineering(self, dataframe_cleaned: pd.DataFrame, new_column_name: str, column_numerator: str, column_denominator: str) -> pd.DataFrame:
        # Create a new feature as a ratio of the specified columns
        dataframe_cleaned[new_column_name] = dataframe_cleaned[column_numerator] / dataframe_cleaned[column_denominator]

        # Split the data into features and target -> These will be later saved in processed data. 
        X = dataframe_cleaned.drop([self.target_column], axis=1)
        y = np.log(dataframe_cleaned[self.target_column])  # Applying log transformation to the target

        skew_df = pd.DataFrame(X.select_dtypes(np.number).columns, columns= ['Feature'])
        skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: skew(X[feature]))
        skew_df['Abs_Skew'] = skew_df['Skew'].apply(abs)
        skew_df['Skewed'] = skew_df['Abs_Skew'].apply(lambda x: True if x > 0.5 else False)
        skew_df
        
        skewed_columns = skew_df[skew_df['Abs_Skew'] > 0.5]['Feature'].values
        skewed_columns

        # Apply log transformation to the skewed columns
        for column in skewed_columns:
            X[column] = np.log1p(X[column])  # Using log1p to handle zero values
        
        return X, y, skew_df
    
    
    @staticmethod
    def encoder_and_scaler(X: pd.DataFrame, column_to_encode: str = None) -> pd.DataFrame:
        
        if column_to_encode is not None:
            encoder = LabelEncoder()
            X[column_to_encode] = encoder.fit_transform(X[column_to_encode])
            
            scaler = StandardScaler()
            scaler.fit(X)
            X = pd.DataFrame(scaler.transform(X), index= X.index, columns= X.columns)
        else:
            scaler = StandardScaler()
            scaler.fit(X)
            X = pd.DataFrame(scaler.transform(X), index= X.index, columns= X.columns)
            
        return X
            
            
    def saving_processed_data(self, X: pd.DataFrame, y: pd.Series, X_name: str, y_name:str) -> None:
        #Creating the directory if doesnt exsist, based on the file structure it should be processed/
        if not os.path.exists(self.folder_name_to_save):
            os.makedirs(self.folder_name_to_save)
            
         # Save X and y to files
        X.to_csv(os.path.join(self.folder_name_to_save, f'{X_name}.csv'), index=False)
        y.to_csv(os.path.join(self.folder_name_to_save, f'{y_name}.csv'), index=False)
        
        print("Processed data saved successfully. :)")
        
        
        
