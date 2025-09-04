import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.components.data_ingestion import DataIngestion

from NetworkSecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig


if __name__ == "__main__":
    try:
        print("Training Pipeline started...")
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        print("Training Pipeline completed successfully.")
    
    except Exception as e:
        NetworkSecurityException(e, sys)
