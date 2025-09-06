import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from NetworkSecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from NetworkSecurity.entity.config_entity import DataTransformationConfig
from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging
from NetworkSecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    """Handles feature engineering and data preprocessing for training and testing datasets."""

    def __init__(
            self,
            data_validation_artifact: DataValidationArtifact,
            data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("Initialized DataTransformation class.")
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

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a preprocessing pipeline with KNNImputer.
        """
        try:
            logging.info("Creating KNNImputer pipeline.")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"KNNImputer initialized with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes the data transformation pipeline:
        - Reads train and test datasets
        - Splits features and target
        - Applies KNNImputer transformation
        - Saves transformed datasets and preprocessing object
        """
        logging.info("Starting data transformation process...")

        try:
            # Load train and test datasets
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split input and target features
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Preprocessing Pipeline
            preprocessor = self.get_data_transformer_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save transformed data
            save_numpy_array_data(
                file_path = self.data_transformation_config.transformed_train_file_path,
                array = train_arr
            )
            save_numpy_array_data(
                file_path = self.data_transformation_config.transformed_test_file_path,
                array = test_arr
            )

            # Save perprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Preprocessor object saved at: {self.data_transformation_config.transformed_object_file_path}")

            # (Optional) Save final model preprocessor (if required in downstream pipeline)
            final_model_preprocessor_path = os.path.join("final_model", "preprocessor.pkl")
            os.makedirs(os.path.dirname(final_model_preprocessor_path), exist_ok=True)
            save_object(
                file_path=final_model_preprocessor_path,
                obj=preprocessor_obj
            )        
            logging.info(f"Final model preprocessor saved at: {final_model_preprocessor_path}")

            # Prepare and return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(data_transformation_artifact)    
            logging.info("Data transformation completed succssfully.\n\n")
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)