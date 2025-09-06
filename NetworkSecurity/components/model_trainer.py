import os
import sys
from typing import Dict, Any
from urllib.parse import urlparse

import mlflow
import dagshub
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from NetworkSecurity.entity.config_entity import ModelTrainerConfig
from NetworkSecurity.utils.ml_utils.model.estimator import NetworkModel
from NetworkSecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from NetworkSecurity.utils.ml_utils.metric.classification_metric import get_classification_score


# Load environment variables
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Initialize DagsHub integration (optional if you already have MLFLOW_TRACKING_URI set)
dagshub.init(repo_owner="thesahilmandal", repo_name="Network-Security-01", mlflow=True)


class ModelTrainer:
    """Handles training, evaluation, MLflow experiment tracking, and saving of ML models."""

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # Setup MLflow tracking
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Network-Security-Model-Training")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _log_to_mlflow(self, model_name: str, model, train_metric, test_metric):
        """
        Logs model metrics, params, and artifact to MLflow.
        """
        try:
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name=f"{model_name}_run"):
                # Log parameters
                mlflow.log_param("model_name", model_name)

                # Log metrics (both train & test)
                mlflow.log_metric("train_f1_score", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)

                mlflow.log_metric("test_f1_score", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # Save model to MLflow
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """Train candidate models, evaluate them, log to MLflow, and save the best-performing one."""
        try:
            # Candidate models
            models: Dict[str, Any] = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000)
            }

            # Hyperparameter grids
            params: Dict[str, Dict[str, Any]] = {
                "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
                "Random Forest": {"n_estimators": [8, 16, 32, 128, 256]},
                "Logistic Regression": {}
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with score {best_model_score}")

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Metrics
            classification_train_metric = get_classification_score(y_train, y_train_pred)
            classification_test_metric = get_classification_score(y_test, y_test_pred)

            # Save preprocessor + model
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor, best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            # Optional: Save raw best model
            save_object("final_model/model.pkl", obj=best_model)

            # Log results to MLflow
            self._log_to_mlflow(best_model_name, best_model, classification_train_metric, classification_test_metric)

            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Load transformed datasets and trigger model training."""
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
