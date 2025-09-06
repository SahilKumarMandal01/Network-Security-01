import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        raise NetworkSecurityException(e, sys)