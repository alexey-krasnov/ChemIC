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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from chemic.chemical_labels import chem_labels
from chemic.config import Config
from chemic.loading_images import MixedImagesDataset
from chemic.utils import get_package_name_version, generate_unique_identifier

# Define the transformation for the images
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=3),  # Convert to RGB if grayscale
    v2.ToImage(),  # Convert to PIL Image
    v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale to [0, 1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageClassifier:
    """
    A class encapsulating image classification functionality.
    """
    def __init__(self) -> None:
        """Initializes the ImageRecognizer instance with queues.
        """
        self.classifier_model = Config().classifier
        self.mixed_loader = None
        self.results = []  # Store results of recognition in a list
        self.classifier_version = f"{get_package_name_version('ChemIC-ml')}"
        self.total_number_images = 0

    def send_to_classifier(self, image_path: str) -> List[dict]:
        """
        Enqueues images for classification based on the provided image path.

        Parameters:
            - image_path (str): Path to the image file or directory

        Returns:
            - List[dict]: List of classification results.
        """
        try:
            # Create a DataLoader for the mixed images
            mixed_dataset = MixedImagesDataset(path_or_dir=image_path, transform=transform)
            self.mixed_loader = DataLoader(mixed_dataset, batch_size=1, shuffle=False, num_workers=0)
            self.total_number_images = +len(self.mixed_loader)
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
            return self.results

    def process_image_files(self) -> None:
        """
        Processes images in the mixed_loader using multithreading.
        """
        with ThreadPoolExecutor(max_workers=min((os.cpu_count()), len(self.mixed_loader))) as executor:
            futures = [executor.submit(self.process_image_file, image_data_) for image_data_ in self.mixed_loader]
            for future in as_completed(futures):
                image_path, predicted_label = future.result()
                print(image_path, predicted_label)
                result_entry = {
                    'image_id': Path(image_path).name,
                    'predicted_label': predicted_label,
                    'classifier_package': self.classifier_version,
                    'classifier_model': f"{self.classifier_model.__class__.__name__}_50",
                }
                self.results.append(result_entry)

    def process_image_file(self, image_data: Tuple[str, torch.Tensor]):
        """
        Processes a single image in the mixed_loader and returns the image path and predicted class label by
        using chemical images classifier.

        Parameters:
        - image_data (Tuple[str, torch.Tensor]): A tuple containing the image path, and the corresponding image tensor.

        Returns:
        - Tuple[str, str]: A tuple containing the image path and the predicted class label.
        """
        image_path, image = image_data
        try:
            predicted_label = self.inference_label(image=image)
            return image_path[0], predicted_label
        except Exception as e:
            # Log the error and re-raise if necessary
            print(f"Error processing image {image_path}: {e}")
            raise e


    def process_image_data(self, base64_data: str) -> List[dict]:
        """
        Processes base64-encoded image data and adds the result to the results list.

        Parameters:
            - base64_data (str): Base64-encoded image data.

        Returns:
            - List[dict]: List of classification results.
        """
        image_hash_id = generate_unique_identifier(base64_encoded_image=base64_data.encode())
        # Transform the image
        transformed_image = self.transform_base64_image(base64_data, transform_type=transform)
        self.total_number_images = +1
        try:
            predicted_label = self.inference_label(image=transformed_image)
        except Exception as e:
            result_entry = {
                'image_id': image_hash_id,
                'predicted_label': f"Error: {str(e)}",
                'classifier_package': self.classifier_version,
                'classifier_model': f"{self.classifier_model.__class__.__name__}_50",
            }
            self.results.append(result_entry)
            return self.results
        else:
            result_entry = {
                'image_id': image_hash_id,  # Consider using a hash of the image data as an ID if needed
                'predicted_label': predicted_label,
                'classifier_package': self.classifier_version,
                'classifier_model': f"{self.classifier_model.__class__.__name__}_50",
            }
            print(f'Result entry: {result_entry}')
            self.results.append(result_entry)
            return self.results

    @staticmethod
    def transform_base64_image(base64_string: str, transform_type) -> torch.Tensor:
        """
        Function to decode base64 string and apply transformations for further prediction with ML model.

        Parameters:
            - base64_string (str): Base64-encoded image data.
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
        # Apply transformations to the image
        transformed_image = transform_type(image).unsqueeze(0)
        return transformed_image
    #
    #
    def inference_label(self, image):
        """
        Performs inference on the image and returns the predicted label.

        Parameters:
           - image: Image tensor.

        Returns:
           - str: Predicted label.
        """
        with torch.no_grad():
            output = self.classifier_model(image)
            _, predicted = torch.max(output.data, 1)
            predicted_label = chem_labels[predicted.item()]
            return predicted_label
