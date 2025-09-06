import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from NetworkSecurity.entity.config_entity import DataValidationConfig
from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from NetworkSecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Performs validation of ingested data included schema checks and drift detection.
    """
    def __init__(
            self,
            data_ingestion_artifact: DataIngestionArtifact,
            data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Initialized DataValidation with schema config loaded.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads a CSV file into a Pandas DataFrame."""
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Validates that the DataFrame has the expected number of columns."""
        try:
            expected_columns = self._schema_config.get('columns', [])
            expected_count = len(expected_columns)

            logging.info(f"Required number of columns: {expected_count}")
            logging.info(f"DataFrame has columns: {len(dataframe.columns)}")

            return len(dataframe.columns) == expected_count
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_datase_drift(
            self,
            base_df: pd.DataFrame,
            current_df: pd.DataFrame,
            threshold: float = 0.05
    ) -> bool:
        """Detects data drift between base and current DataFrames using KS-test."""
        try:
            status = True
            report = {}

            logging.info("Starting dataset drift detection...")
            for column in base_df.columns:
                d1, d2 = base_df[column], current_df[column]
                ks_test = ks_2samp(d1, d2)

                drift_found = ks_test.pvalue < threshold
                if drift_found:
                    status = False

                report[column] = {
                    "p_value": float(ks_test.pvalue),
                    "drift_status": drift_found
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            logging.info(f"Drift detection completed. Report saved at: {drift_report_file_path}")
            return status
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Excecuted full data validation workflow: schema validation + drift detection."""
        logging.info("Starting Data Validation...")
        try:
            train_fle_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Load train and test data
            train_df = self.read_data(train_fle_path)
            test_df = self.read_data(test_file_path)

            # Validate number of columns
            if not self.validate_number_of_columns(train_df):
                raise NetworkSecurityException("Train dataframe does not contain all required columns.", sys)
            
            if not self.validate_number_of_columns(test_df):
                raise NetworkSecurityException("Test dataframe does not contain all required columns.", sys)
            
            # Check for drift
            validation_status = self.detect_datase_drift(train_df, test_df)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_test_file_path=None,
                invalid_train_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(data_validation_artifact)
            logging.info("Data validation completed successfully.\n\n")

            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)