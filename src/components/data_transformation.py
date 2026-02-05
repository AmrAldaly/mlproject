import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.base import TransformerMixin ,BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,StandardScaler
from scipy.stats.mstats import winsorize

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class Winsorizer:
    """Custom Winsorizer transformer compatible with sklearn Pipeline"""
    def __init__(self, limits=[0, 0.02]):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = winsorize(X[col].values, limits=self.limits)
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Columns
            robust_cols = ["creatinine_phosphokinase", "serum_creatinine", "platelets"]
            standard_cols = ["age", "ejection_fraction", "serum_sodium", "time"]

            # Pipelines
            robust_pipeline = Pipeline([
                ("winsorize", Winsorizer(limits=[0,0.02])),
                ("robust_scaler", RobustScaler())
            ])
            standard_pipeline = Pipeline([
                ("standard_scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("standard_pipeline", standard_pipeline, standard_cols),
                ("robust_pipeline", robust_pipeline, robust_cols)
            ], remainder="passthrough")  # keep binary columns as is

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()
            target_col = "DEATH_EVENT"

            # Separate features & target
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # Fit transform
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            # Combine back with target for consistency
            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            # Save preprocessor
            save_object(file_path=self.config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)