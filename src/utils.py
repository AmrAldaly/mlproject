import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import f1_score


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def predict_with_threshold(model, X, threshold=0.25):
    try:
        probabilities = model.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    except Exception as e:
        raise CustomException(e, sys)