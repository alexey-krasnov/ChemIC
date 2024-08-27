"""Chemical Images Classifier Module:
This module provides image classification functionality.
It uses pre-trained models for classifying chemical images.

Dependencies:
    - concurrent
    - pathlib
    - typing
    - flask
    - torch
    - torchvision
    - config (assuming Config class is defined in the 'config' module)
    - loading_images (assuming MixedImagesDataset class is defined in the 'loading_images' module)

Usage:
    1. Instantiate the ImageClassifier class.
        classifier = ImageClassifier()

    2. Using image path or directory:
        results = classifier.send_to_classifier(image_path_or_dir)

    3. Using base64-encoded image data:

        base64_data = <class 'bytes'>  # Replace with your base64-encoded image data
        results = classifier.process_image_data(base64_data)

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: February 26, 2024
"""

import base64
import importlib.metadata
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, NamedTuple, Union

import torch
from PIL import Image
from flask import jsonify, Response
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from chemic.config import Config
from chemic.loading_images import MixedImagesDataset

# Define the transformation for the images
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=3),  # Convert to RGB if grayscale
    # v2.ToTensor(), # will be removed in a future release. Instead we are using next 2 lines:
    v2.ToImage(),  # Convert to PIL Image
    v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define class labels. Order of class label in the NamedTuple is essential!
class ChemicalLabels(NamedTuple):
    """"Class label for image classifier"""
    single_chemical_structure: str
    chemical_reactions: str
    no_chemical_structures: str
    multiple_chemical_structures: str


# Creating an instance of ChemicalLabels
chem_labels = ChemicalLabels(single_chemical_structure='single chemical structure',
                             chemical_reactions='chemical reactions',
                             no_chemical_structures='no chemical structures',
                             multiple_chemical_structures='multiple chemical structures')

# Load ML models
classifier_model = Config.get_models()


class ImageClassifier:
    """
    A class encapsulating image classification functionality.
    """
    def __init__(self) -> None:
        """Initializes the ImageRecognizer instance with queues.
        """
        self.mixed_loader = None
        self.results = []  # Store results of recognition in a list
        self.flag = 'ChemIC-ml'
        self.version = self.get_package_version(self.flag)

    def send_to_classifier(self, image_path: str) -> Union[Tuple[Response, int], List]:
        """
        Enqueues images for classification based on the provided image input.

        Parameters:
            - image_input (str): Path to the image file or directory

        Returns:
            - Union[Tuple[Response, int], List]: Tuple containing success message or error response if input is invalid,
                                                 or list of classification results.
        """
        try:
            # Create a DataLoader for the mixed images
            mixed_dataset = MixedImagesDataset(path_or_dir=image_path, transform=transform)
            self.mixed_loader = DataLoader(mixed_dataset, batch_size=1, shuffle=False, num_workers=0)
        except Exception as e:
            print(f"Exception: {e} {image_path}")
            result_entry = {
                'image_id': image_path,
                'predicted_label': 'Error! File is not an image',
            }
            self.results.append(result_entry)
            return self.results
        else:
            # Perform classification
            self.process_image_files()
            return jsonify({"message": "Images have been classified."}), 202

    def process_image_files(self) -> None:
        """
        Processes images in the mixed_loader using multithreading.
        The images are processed concurrently using a ThreadPoolExecutor with a maximum number of worker threads
        determined by min of the CPU count or number of images in self.mixed_loader.
        """
        with ThreadPoolExecutor(max_workers=min((os.cpu_count()), len(self.mixed_loader))) as executor:
            futures = [executor.submit(self.process_image_file, image_data_) for image_data_ in self.mixed_loader]
            for future in as_completed(futures):
                image_path, predicted_label = future.result()
                print(image_path, predicted_label)
                result_entry = {
                    'image_id': Path(image_path).name,
                    'predicted_label': predicted_label,
                    'program': self.flag,
                    'program_version': self.version
                }
                self.results.append(result_entry)

    def process_image_file(self, image_data: Tuple[str, torch.Tensor]):
        """
        Processes a single image in the mixed_loader and returns the image path and predicted class label by
        using chemical images classifier.

        Parameters:
        - image_data (Tuple[str, torch.Tensor]): A tuple containing the image path and the corresponding image tensor.

        Returns:
        - Tuple[str, str]: A tuple containing the image path and the predicted class label.
        """
        image_path, image = image_data
        image_path = image_path[0]  # Extract the image path from the batch
        try:
            predicted_label = self.inference_label(image=image)
            return image_path, predicted_label
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    def process_image_data(self, base64_data):
        """
        Processes base64-encoded image data and adds the result to the results list.

        Parameters:
            - base64_data (str): Base64-encoded image data.
        """
        # Transform the image
        transformed_image = self.transform_base64_image(base64_data, transform_type=transform)

        try:
            predicted_label = self.inference_label(image=transformed_image)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        else:
            result_entry = {
                'image_id': None,  # TODO: should we use hash of binary object to identify it or just skip image_id?
                'predicted_label': predicted_label,
                'program': self.flag,
                'program_version': self.version
            }
            print(f'Result entry {result_entry}')
            self.results.append(result_entry)
            return self.results

    @staticmethod
    def transform_base64_image(base64_string, transform_type):
        """
        Function to decode base64 string and apply transformations for further prediction with ML model.

        Parameters:
            - base64_data (str): Base64-encoded image data.
            - transform_type: Transformation to apply to the image.

        Returns:
            - torch.Tensor: Transformed image tensor ready for prediction.
        """
        # Decode the base64 encoded image data
        decoded_data = base64.b64decode(base64_string)
        # Create a BytesIO object from the decoded binary data
        image_stream = BytesIO(decoded_data)
        # Open the image using PIL.Image.open()
        image = Image.open(image_stream)
        # Apply transformations for images
        transformed_image = transform_type(image).unsqueeze(0)
        return transformed_image

    @staticmethod
    def inference_label(image):
        """
        Performs inference on the image and returns the predicted label.

        Parameters:
           - image: Image tensor.

        Returns:
           - str: Predicted label.
        """
        with torch.no_grad():
            output = classifier_model(image)
            _, predicted = torch.max(output.data, 1)
            predicted_label = chem_labels[predicted.item()]
            return predicted_label

    @staticmethod
    def get_package_version(package_name):
        """
        Get the version of the specified Python package.

        Parameters:
            package_name (str): The name of the Python package.

        Returns:
            str: The version of the specified package if installed.
                 If the package is not installed, returns a message indicating that the package is not installed.
        """
        try:
            package_version = importlib.metadata.version(package_name)
            return package_version
        except importlib.metadata.PackageNotFoundError:
            return f"{package_name} is not installed"
