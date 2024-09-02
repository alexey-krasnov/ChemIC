"""Configuration Module:
This module provides configuration settings for the image classification application.

Dependencies:
    - torch
    - torchvision
    - pathlib

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: February 26, 2024
"""
import io
import zipfile
from functools import lru_cache
from pathlib import Path

import requests
import torch
from torchvision import models

# Get the absolute path of the current file's directory
CURRENT_DIR = Path(__file__).resolve().parent

class Config:
    # Adjust the path to point to the 'models' directory relative to the current file's directory
    MODELS_DIR = CURRENT_DIR / 'models'

    IMAGE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "chemical_image_classifier_resnet50.pth"

    # IMAGE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "chemical_image_classifier_resnet50_29epochs_hand_drawn_like_2024-02-27T15:56:37.pth"

    PROCESSING_UNIT = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    API_URL = 'http://127.0.0.1:5010'

    def __init__(self):
        """
        Initializes the Config instance, setting up placeholders for model attributes.
        """
        self._classifier = None

    @property
    def classifier(self):
        """
        Returns the classifier model, initializing it if necessary.

        Returns:
            torch.nn.Module: The initialized classifier model.
        """
        if self._classifier is None:
            self.init_classifier()
        return self._classifier

    @lru_cache(maxsize=None)
    def init_classifier(self):
        """
        Initializes the classifier model.

        If the model file does not exist locally, it will be downloaded and extracted.
        The model is then loaded and prepared for inference.

        Returns:
            torch.nn.Module: The initialized classifier model.
        """
        if not Config.IMAGE_CLASSIFIER_MODEL_PATH.exists():
            print(f'Downloading classifier models from Zenodo..')
            Config.download_and_extract_chemic_model()
        model = models.resnet50(pretrained=False)
        num_classes = 4
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(Config.IMAGE_CLASSIFIER_MODEL_PATH))
        self._classifier = model.eval()

    @staticmethod
    def download_and_extract_chemic_model():
        """
        Downloads and extracts the chemical models archive from a predefined URL.

        This method downloads a ZIP archive containing the model files and extracts it
        to the specified models directory.
        """
        url = "https://zenodo.org/record/10709886/files/models.zip"
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(Config.MODELS_DIR)
