import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation

from NetworkSecurity.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig,
    DataValidationConfig
)
from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


if __name__ == "__main__":
    try:
        print("Training Pipeline started...")
        
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        
        print("Training Pipeline completed successfully.")
    
    except Exception as e:
        NetworkSecurityException(e, sys)
