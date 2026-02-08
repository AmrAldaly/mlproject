import os
import sys
from dataclasses import dataclass


from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Split features & target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=4,
                max_iter=100,
                l2_regularization=0.1,
                class_weight='balanced',
                random_state=42
            )

            model.fit(X_train, y_train)

            # Custom threshold 
            probabilities = model.predict_proba(X_test)[:, 1]
            threshold = 0.25
            y_pred_custom = (probabilities >= threshold).astype(int)

            # Metrics
            logging.info("Classification Report:\n" + str(classification_report(y_test, y_pred_custom)))

            # Save model
            save_object(file_path=self.config.trained_model_file_path, obj=model)

            f1 = f1_score(y_test, y_pred_custom)
            return f1

        except Exception as e:
            raise CustomException(e, sys)