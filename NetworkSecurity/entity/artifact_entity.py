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