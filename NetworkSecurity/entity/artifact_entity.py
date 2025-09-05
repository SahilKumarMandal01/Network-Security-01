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