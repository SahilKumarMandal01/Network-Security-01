import os
import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation
from NetworkSecurity.components.data_transformation import DataTransformation
from NetworkSecurity.components.model_trainer import ModelTrainer

from NetworkSecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from NetworkSecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)


class TrainingPipeline:
    """Orchestrates the entire ML pipeline: ingestion → validation → transformation → training."""

    def __init__(self) -> None:
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            logging.info("✅ TrainingPipeline initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ------------------------------
    # 🔹 Pipeline Stages
    # ------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("🚀 Starting Data Ingestion...")
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data Ingestion completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("🚀 Starting Data Validation...")
            config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=config,
            )
            artifact = data_validation.initiate_data_validation()
            logging.info(f"✅ Data Validation completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("🚀 Starting Data Transformation...")
            config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=config,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info(f"✅ Data Transformation completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("🚀 Starting Model Training...")
            config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=config,
            )
            artifact = model_trainer.initiate_model_trainer()
            logging.info(f"✅ Model Training completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ------------------------------
    # 🔹 Orchestration
    # ------------------------------
    def run_pipeline(self) -> ModelTrainerArtifact:
        """Run the full ML pipeline from start to finish."""
        try:
            logging.info("🔥 Pipeline execution started...")
            
            # Run sequential stages
            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)
            transformation_artifact = self.start_data_transformation(validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)

            # Sync artifacts & models to cloud (if implemented)
            if hasattr(self, "sync_artifact_dir_to_s3"):
                self.sync_artifact_dir_to_s3()
            if hasattr(self, "sync_saved_model_dir_to_s3"):
                self.sync_saved_model_dir_to_s3()

            logging.info("✅ Pipeline execution completed successfully.")
            return trainer_artifact

        except Exception as e:
            logging.error("❌ Pipeline execution failed.")
            raise NetworkSecurityException(e, sys)
