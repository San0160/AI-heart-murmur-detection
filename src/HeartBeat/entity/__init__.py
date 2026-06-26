from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen = True)
class DataingestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen = True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: Path

@dataclass(frozen = True)
class DataPreprocessingConfig:
    root_dir: Path
    local_data_file: Path

@dataclass(frozen = True)
class DataTransformerConfig:
    root_dir: Path
    local_data_file: Path
    preprocessed_data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_data_path: Path
    trained_model_path: Path

    input_size: int
    hidden_size: int
    num_layers: int
    num_classes: int
    dropout: float

    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    weight_decay: float