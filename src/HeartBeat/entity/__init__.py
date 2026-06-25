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