import yaml
from networksecurity.exception.exception import NetworkSecurityException
import sys, os
from networksecurity.logging.logger import logging
import numpy as np
import pandas as pd
import pickle
# import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
import math

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(
    X_train, y_train, X_test, y_test,
    models: Dict[str, Any],
    params: Optional[Dict[str, Dict]] = None,
    problem_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate provided models on train and test sets.
    - models: dict of name->estimator
    - params: dict of name->param_grid for GridSearchCV (optional)
    - problem_type: 'classification' or 'regression' (auto-detected if None)

    Returns a report dict:
    {
      model_name: {
        "best_params": {...},
        "train": {...metrics...},
        "test": {...metrics...}
      },
      ...
    }
    """
    try:
        report = {}

        # auto-detect problem type if not provided
        if problem_type is None:
            unique_vals = getattr(y_train, "nunique", lambda: None)()
            if unique_vals is not None and unique_vals <= 20 and (str(y_train.dtype).startswith("int") or str(y_train.dtype).startswith("object")):
                prob_type = "classification"
            else:
                prob_type = "regression"
        else:
            prob_type = problem_type.lower()

        for name, estimator in models.items():
            model = estimator
            best_params = {}

            # hyperparameter search if grid provided for this model
            if params and name in params and params[name]:
                gs = GridSearchCV(model, params[name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_params = gs.best_params_
                model.set_params(**best_params)

            # fit model
            model.fit(X_train, y_train)

            # predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            model_report = {"best_params": best_params, "train": {}, "test": {}}

            if prob_type == "classification":
                # decide average for multiclass vs binary
                average = "binary" if (getattr(y_train, "nunique", lambda: 2)() == 2) else "weighted"

                model_report["train"]["accuracy"] = accuracy_score(y_train, y_train_pred)
                model_report["train"]["precision"] = precision_score(y_train, y_train_pred, average=average, zero_division=0)
                model_report["train"]["recall"] = recall_score(y_train, y_train_pred, average=average, zero_division=0)
                model_report["train"]["f1"] = f1_score(y_train, y_train_pred, average=average, zero_division=0)

                model_report["test"]["accuracy"] = accuracy_score(y_test, y_test_pred)
                model_report["test"]["precision"] = precision_score(y_test, y_test_pred, average=average, zero_division=0)
                model_report["test"]["recall"] = recall_score(y_test, y_test_pred, average=average, zero_division=0)
                model_report["test"]["f1"] = f1_score(y_test, y_test_pred, average=average, zero_division=0)

                # try ROC AUC when predict_proba or decision_function available (binary or multiclass with average="weighted")
                try:
                    if hasattr(model, "predict_proba"):
                        y_test_proba = model.predict_proba(X_test)
                        # binary
                        if y_test_proba.shape[1] == 2:
                            model_report["test"]["roc_auc"] = roc_auc_score(y_test, y_test_proba[:, 1])
                        else:
                            model_report["test"]["roc_auc"] = roc_auc_score(y_test, y_test_proba, multi_class="ovr", average=average)
                    elif hasattr(model, "decision_function"):
                        scores = model.decision_function(X_test)
                        model_report["test"]["roc_auc"] = roc_auc_score(y_test, scores)
                except Exception:
                    # ignore ROC AUC calculation failures
                    pass

            else:  # regression
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)

                model_report["train"]["r2"] = train_r2
                model_report["train"]["rmse"] = train_rmse
                model_report["train"]["mae"] = train_mae

                model_report["test"]["r2"] = test_r2
                model_report["test"]["rmse"] = test_rmse
                model_report["test"]["mae"] = test_mae

            report[name] = model_report

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e