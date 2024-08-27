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
from pathlib import Path

import requests
import torch
from torchvision import models


class Config:
    # Get the absolute path of the current file's directory
    CURRENT_DIR = Path(__file__).resolve().parent
    # Adjust the path to point to the 'models' directory relative to the current file's directory
    MODELS_DIR = CURRENT_DIR / 'models'

    IMAGE_CLASSIFIER_MODEL_PATH = MODELS_DIR / "chemical_image_classifier_resnet50.pth"

    PROCESSING_UNIT = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def download_and_extract_chemic_model():
        # Download the models.zip file from Zenodo
        url = "https://zenodo.org/record/10709886/files/models.zip"
        response = requests.get(url)
        response.raise_for_status()

        # Extract the contents of the models.zip file
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(Config.MODELS_DIR)

    @staticmethod
    def get_classifier_model():
        if not Config.IMAGE_CLASSIFIER_MODEL_PATH.exists():
            # Download and extract models if not already downloaded
            print(f'Downloading models...')
            Config.download_and_extract_chemic_model()

        model = models.resnet50(pretrained=False)  # False if not using a pretrained model from pytorch
        num_classes = 4  # Adjust the number of classes in your model
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(Config.IMAGE_CLASSIFIER_MODEL_PATH))
        return model.eval()

    @staticmethod
    def get_models():
        return Config.get_classifier_model()
