from pathlib import Path

from huggingface_hub import hf_hub_download

from HeartBeat.config.configuration import PredictionConfig


class ModelDownloader:
    def __init__(self, config: PredictionConfig):
        self.config = config

    def download_if_missing(self):
        """
        Download the trained model from Hugging Face if it
        does not already exist locally.
        """

        model_path = Path(self.config.trained_model_path)

        # Model already exists
        if model_path.exists():
            print("✓ Trained model already exists.")
            return model_path

        print("Downloading trained model from Hugging Face...")

        # Create folder if missing
        model_path.parent.mkdir(parents=True, exist_ok=True)

        hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename,
            local_dir=model_path.parent,
            local_dir_use_symlinks=False
        )

        print("✓ Model downloaded successfully.")

        return model_path