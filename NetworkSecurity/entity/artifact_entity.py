from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

    def __str__(self):
        return (
            f"\nDataIngestionArtifact:\n"
            f"  Trained File Path: {self.trained_file_path}\n"
            f"  Test File Path   : {self.test_file_path}\n"
        )


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

    def __str__(self):
        return (
            f"\nDataValidationArtifact:\n"
            f"  Validation Status      : {self.validation_status}\n"
            f"  Valid Train File Path  : {self.valid_train_file_path}\n"
            f"  Valid Test File Path   : {self.valid_test_file_path}\n"
            f"  Invalid Train File Path: {self.invalid_train_file_path}\n"
            f"  Invalid Test File Path : {self.invalid_test_file_path}\n"
            f"  Drift Report File Path : {self.drift_report_file_path}\n"
        )


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

    def __str__(self):
        return (
            f"\nDataTransformationArtifact:\n"
            f"  Transformed Object File Path: {self.transformed_object_file_path}\n"
            f"  Transformed Train File Path : {self.transformed_train_file_path}\n"
            f"  Transformed Test File Path  : {self.transformed_test_file_path}\n"
        )


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

    def __str__(self):
        return (
            f"\n    ClassificationMetricArtifact:\n"
            f"         F1 score       : {self.f1_score:.4f}\n"
            f"         Precision Score: {self.precision_score:.4f}\n"
            f"         Recall Score   : {self.recall_score:.4f}\n"
        )


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

    def __str__(self):
        return (
            f"\nModelTrainerArtifact:\n"
            f"  Trained Model File Path: {self.trained_model_file_path}\n"
            f"  Train Metrics          : {self.train_metric_artifact}\n"
            f"  Test Metrics           : {self.test_metric_artifact}\n"
        )