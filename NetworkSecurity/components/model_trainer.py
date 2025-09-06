import os
import sys
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from NetworkSecurity.entity.config_entity import ModelTrainerConfig
from NetworkSecurity.entity.config_entity import ModelTrainerConfig
from NetworkSecurity.utils.ml_utils.model.estimator import NetworkModel
from NetworkSecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from NetworkSecurity.utils.ml_utils.metric.classification_metric import get_classification_score


class ModelTrainer:
    """Handles training, evaluation, and saving of ML models for network security."""

    def __init__(
            self,
            model_trainer_config: ModelTrainerConfig,
            data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def train_model(self, X_train, y_train, X_test, y_test) -> ModelTrainerArtifact:
        """Train candidates models, evaluate them, and save the best-performing one."""
        try:
            # Candiate models
            models: Dict[str, Any] = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000) 
            }

            # Hyperparameters grids
            params: Dict[str, Dict[str, Any]] = {
                "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
                "Random Forest": {"n_estimators": [8, 16, 32, 128, 256]},
                "Logistic Regression": {}
            }

            # Evaluate all models
            logging.info("Started Model Training...")
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model selected: {best_model_name} with score {best_model_score}")
            logging.info("Model Training completed succesfully.")
            
            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Metrics
            classification_train_metric = get_classification_score(y_train, y_train_pred)
            classification_test_metrics = get_classification_score(y_test, y_test_pred)

            # Load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # Save trained model
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor, best_model)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            # Optional: Save only the raw best model
            save_object("final_model/model.pkl", obj=best_model)

            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metrics
            )
            
            logging.info(model_trainer_artifact)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Load transformed datasets and trigger model training."""
        try:
            # Load training and testing arrays
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
