import os, sys
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkSecurityModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def track_mlflow(self, model, metrics: dict):
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

    def train_model(self, X_train, y_train, X_test, y_test) -> NetworkSecurityModel:
        try:
            models = {
                "LogisticRegression": LogisticRegression(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier()
            }

            # param grids keyed to the same model names used above
            params = {
                "DecisionTreeClassifier": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "RandomForestClassifier": {
                    "n_estimators": [8, 16, 32, 128, 256]
                },
                "GradientBoostingClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "LogisticRegression": {},
                "AdaBoostClassifier": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "KNeighborsClassifier": {}
            }

            # evaluate models (this will run GridSearch where params provided)
            report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params, problem_type="classification"
            )
            logging.info(f"Model evaluation report: {report}")

            # choose best model by test f1 (fallback to accuracy)
            best_model_name = None
            best_score = -float("inf")
            for name, result in report.items():
                test_metrics = result.get("test", {})
                score = test_metrics.get("f1") or test_metrics.get("accuracy") or test_metrics.get("r2") or 0
                if score > best_score:
                    best_score = score
                    best_model_name = name

            if best_model_name is None:
                raise Exception("No candidate models found in evaluation report.")

            logging.info(f"Best model: {best_model_name} with score: {best_score}")

            # enforce expected accuracy threshold
            if best_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No model met the expected accuracy: {self.model_trainer_config.expected_accuracy}. "
                    f"Best score: {best_score} (model: {best_model_name})"
                )

            # prepare and fit the selected model using best params (if any)
            best_model = models[best_model_name]
            best_params = report[best_model_name].get("best_params", {})
            if best_params:
                best_model.set_params(**best_params)

            # fit selected model
            best_model.fit(X_train, y_train)

            # evaluate fitted model using get_classification_score
            try:
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_metrics = get_classification_score(y_train, y_train_pred)
                test_metrics = get_classification_score(y_test, y_test_pred)

                self.track_mlflow(best_model, train_metrics)
                self.track_mlflow(best_model, test_metrics)

                logging.info(
                    f"Selected model '{best_model_name}' metrics -> "
                    f"Train: f1={train_metrics.f1_score:.4f}, precision={train_metrics.precision_score:.4f}, recall={train_metrics.recall_score:.4f}; "
                    f"Test: f1={test_metrics.f1_score:.4f}, precision={test_metrics.precision_score:.4f}, recall={test_metrics.recall_score:.4f}"
                )
            except Exception as e:
                logging.warning(f"Failed to compute classification metrics for model '{best_model_name}': {e}")

            # load preprocessor/transform object and wrap model
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkSecurityModel(model=best_model, preprocessor=preprocessor)

            # persist trained wrapped model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)
            #model pusher
            save_object("final_model/model.pkl",best_model)

            ## Model Trainer Artifact
            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=train_metrics,
                             test_metric_artifact=test_metrics
                             )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try: 
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading transformed training data from: {train_file_path}")
            train_array = load_numpy_array_data(file_path=train_file_path)
            test_array = load_numpy_array_data(file_path=test_file_path)

            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            model_trainer_artifact=self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e