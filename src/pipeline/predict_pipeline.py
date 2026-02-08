import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
        age: float,
        anaemia: int,
        creatinine_phosphokinase: float,
        diabetes: int,
        ejection_fraction: float,
        high_blood_pressure: int,
        platelets: float,
        serum_creatinine: float,
        serum_sodium: float,
        sex: int,
        smoking: int,
        time: float
    ):

        self.age = age
        self.anaemia = anaemia
        self.creatinine_phosphokinase = creatinine_phosphokinase
        self.diabetes = diabetes
        self.ejection_fraction = ejection_fraction
        self.high_blood_pressure = high_blood_pressure
        self.platelets = platelets
        self.serum_creatinine = serum_creatinine
        self.serum_sodium = serum_sodium
        self.sex = sex
        self.smoking = smoking
        self.time = time

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "age": [self.age],
                "anaemia": [self.anaemia],
                "creatinine_phosphokinase": [self.creatinine_phosphokinase],
                "diabetes": [self.diabetes],
                "ejection_fraction": [self.ejection_fraction],
                "high_blood_pressure": [self.high_blood_pressure],
                "platelets": [self.platelets],
                "serum_creatinine": [self.serum_creatinine],
                "serum_sodium": [self.serum_sodium],
                "sex": [self.sex],
                "smoking": [self.smoking],
                "time": [self.time],
            })
        except Exception as e:
            raise CustomException(e, sys)
0