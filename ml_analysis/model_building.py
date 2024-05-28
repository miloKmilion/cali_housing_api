import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import pickle

@dataclass
class LoadDataForML:
    folder_to_save_model: str
    random_state: int
    test_size: float
    
    def create_train_test_split(self, X, y) -> None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        # print the sizes of the resulting data splits
        print("X Training set size:", len(X_train))
        print("X Test set size:", len(X_test))
        print("y train set size:", len(y_train))
        print("y Test set size:", len(y_test))

        return X_train, X_test, y_train, y_test
    
    
    def model_tester(self, X_train, y_train, X_test, y_test):
        # List of regression models to test
        models_to_test = [
            ('Linear Regression', LinearRegression()),
            ('Random Forest', RandomForestRegressor(random_state=self.random_state)),
            ('CatBoost', CatBoostRegressor(verbose=0, random_state=self.random_state)),
            ('XGBoost', XGBRegressor(random_state=self.random_state)),
            ('LightGBM', LGBMRegressor(random_state=self.random_state)),
            ('Gradient Boosting', GradientBoostingRegressor(random_state=self.random_state))
        ]

        results = []
        names = []
        predictions = {}
        scores = {}
        
        print("\nResults:")
        for name, model in models_to_test:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)

            results.append(r2)
            names.append(name)
            scores[name] = r2

            print(f"{name}: RMSE = {rmse}, R-squared = {r2}")

        self.plotting_model_tester(results, names)

        return predictions, y_test, scores
    

    def ensemble_selecting(self, predictions, y_test, scores, threshold=0.80):
        selected_models = [name for name, score in scores.items() if score >= threshold]
        print(f"Selected models for ensemble: {selected_models}")

        if selected_models:
            ensemble_preds = np.mean([predictions[name] for name in selected_models], axis=0)

            # Exponentiate the predictions to return to the original scale
            ensemble_preds_exp = np.exp(ensemble_preds)
            y_test_exp = np.exp(y_test)

            ensemble_rmse = np.sqrt(mean_squared_error(y_test_exp, ensemble_preds_exp))
            ensemble_r2 = r2_score(y_test_exp, ensemble_preds_exp)

            print(f"\nAveraging Ensemble: RMSE = {ensemble_rmse}, R-squared = {ensemble_r2}")

            # Save the ensemble model
            with open(self.folder_to_save_model + 'model.pkl', 'wb') as f:
                pickle.dump({name: predictions[name] for name in selected_models}, f)
            
            return y_test_exp, ensemble_preds_exp
        else:
            print("No models selected for ensemble.")
            return None
    
    def single_model_save(self, X_train, y_train, X_test, y_test, pickle_file):
        model = CatBoostRegressor(verbose=0, random_state=self.random_state)
        print(model)
        # Fit the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        test_score = model.score(X_test, y_test)
        print("Test score:", test_score)
        
        # Save the model to a pickle file
        with open(self.folder_to_save_model + f'{pickle_file}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print("Model saved to", pickle_file)
    
    @staticmethod
    def plotting_model_tester(results, names):
        # Plotting R-squared scores
        plt.figure(figsize=(10, 5))
        sns.barplot(x=names, y=results)
        plt.xlabel('Algorithm')
        plt.ylabel('R-squared')
        plt.title('Comparison of Regression Algorithms')
        plt.xticks(rotation=45)
        plt.show()
        

    def plot_predictions_vs_actual(self, y_test_exp, ensemble_preds_exp):
        y_test_exp = y_test_exp.squeeze()
        # Create DataFrame to compare actual and predicted values
        comparison_df = pd.DataFrame({'Actual': y_test_exp, 'Predicted': ensemble_preds_exp.round(2)})

        # Scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_exp, y=ensemble_preds_exp, color='#005b96')
        plt.xlabel('Actual House Value')
        plt.ylabel('Predicted House Value')
        plt.title('Actual vs Predicted House Values')
        plt.show()

        # Residual plot
        resid = y_test_exp - ensemble_preds_exp
        plt.figure(figsize=(10, 6))
        sns.histplot(resid, kde=True)
        plt.xlabel('Error')
        plt.title('Distribution of Prediction Errors')
        plt.show()

        return comparison_df

    
        
        
    
    