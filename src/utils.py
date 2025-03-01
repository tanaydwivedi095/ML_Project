import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        report = {}
        for item in model.items():
            model_name = item[0]
            model = item[1]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = {
                'train_model_score': train_model_score,
                'test_model_score': test_model_score,
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)

def get_best_model_name(model_report):
    max_test_score = 0
    best_model_name = None
    for item in model_report.items():
        if item[1]['test_model_score'] > max_test_score:
            max_test_score = item[1]['test_model_score']
            best_model_name = item[0]
    return best_model_name