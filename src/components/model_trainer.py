import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, get_best_model_name

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initialize_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Cat Boost": CatBoostRegressor(),
                "XG Boost": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "K Nearest Neighbors": KNeighborsRegressor(),
            }
            model_report: dict = evaluate_model(X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                models)
            best_model_name = get_best_model_name(model_report)
            best_model = models[best_model_name]
            logging.info(f"Best model name: {best_model_name}")
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Model saved")
            predicted = best_model.predict(X_test)
            logging.info("Prediction Done")
            model_r2_score = r2_score(y_test, predicted)
            logging.info("Model R2 Score: {}".format(model_r2_score))
            return model_r2_score
        except Exception as e:
            raise CustomException(e, sys)