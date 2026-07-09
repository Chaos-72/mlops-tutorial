import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import warnings

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        """
        This function train and evaluate the models and pick the best model based
        """
        try:
            logging.info("Initiate model trainer")
            logging.info("Split the training and test dataset")
            
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], # keep the all features except last one (target feature)
                train_arr[:, -1], # Keep the last feature only(target feature)
                # Similarly for test set
                test_arr[:, :-1],
                test_arr[:, -1] 
            )

            # print(f"Shape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_test: {y_test.shape}\n")

            # print(f"\n\nFeature:\nX_train = {X_train.column}\ny_train = {y_train.column}\nX_test = {X_test.column}\ny_test = {y_test.column}")

            models = {
                "Linear Regression": LinearRegression(),
                "XGBoost Regressor": XGBRegressor(),
                "SVM": SVR(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "K Nearest Neighbors": KNeighborsRegressor(),
                "Ridge Regresion": Ridge(),
                "Lasso Regression": Lasso(),
                "CatBoost Regressor": CatBoostRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Highest r2 score among all models
            best_model_score = max(model_report.values())

            # To get the best model name from the dictionay
            best_model_name =list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # To get the best model constructor
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No model found.")
            
            logging.info("Best model found on training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                object = best_model
            )

            predicted = best_model.predict(X_test)
            r_squared = r2_score(y_test, predicted)   

            return r_squared
                
        except Exception as e:
            raise CustomException(e, sys)