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

from NetworkSecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
from NetworkSecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    """Orchestrates the entire ML pipeline: ingestion → validation → transformation → training."""

    def __init__(self) -> None:
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
            self.s3_sync = S3Sync()
            logging.info("✅ TrainingPipeline initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ------------------------------
    # 🔹 Pipeline Stages
    # ------------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("🚀 Starting Data Ingestion...")
            config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            ingestion = DataIngestion(data_ingestion_config=config)
            artifact = ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data Ingestion completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("🚀 Starting Data Validation...")
            config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            validation = DataValidation(
                data_ingestion_artifact=ingestion_artifact,
                data_validation_config=config,
            )
            artifact = validation.initiate_data_validation()
            logging.info(f"✅ Data Validation completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("🚀 Starting Data Transformation...")
            config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            transformation = DataTransformation(
                data_validation_artifact=validation_artifact,
                data_transformation_config=config,
            )
            artifact = transformation.initiate_data_transformation()
            logging.info(f"✅ Data Transformation completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("🚀 Starting Model Training...")
            config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            trainer = ModelTrainer(
                data_transformation_artifact=transformation_artifact,
                model_trainer_config=config,
            )
            artifact = trainer.initiate_model_trainer()
            logging.info(f"✅ Model Training completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ------------------------------
    # 🔹 Cloud Sync
    # ------------------------------
    def sync_artifact_dir_to_s3(self):
        """Sync local artifacts directory to S3."""
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )
            logging.info(f"☁️ Artifacts synced to S3 at: {aws_bucket_url}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        """Sync local trained models to S3."""
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.model_dir,
                aws_bucket_url=aws_bucket_url,
            )
            logging.info(f"☁️ Final models synced to S3 at: {aws_bucket_url}")
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

            # Sync artifacts & models to S3
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            logging.info("✅ Pipeline execution completed successfully.")
            return trainer_artifact

        except Exception as e:
            logging.error("❌ Pipeline execution failed.")
            raise NetworkSecurityException(e, sys)
