import os
import sys

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd 
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object 

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path_file = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformation pipeline")

            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('oh_encoder', OneHotEncoder()),
                    # ('scalar', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns) 
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test set completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            numerical_columns = ['reading_score', 'writing_score']
            target_column = 'math_score'

            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Using preprocessing object on train and test dataset")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            # combine the preprocessed input feature and target feature into a train array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            # combine the preprocessed input feature and target feature into a test array
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path_file,
                object = preprocessing_obj
            )

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_path_file
            )
        
        except Exception as e:
            raise CustomException(e, sys)