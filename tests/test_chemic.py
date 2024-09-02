"""
test_chemic.py

This module contains unit tests for the ChemICR application, focusing on health checks and chemical recognition.

Usage:
    - Run the tests using a testing framework pytest:
        $ pytest test_chemic.py

    - The script sends a POST request to the Flask app,
    which then processes the image and returns the predicted SMILES string.

Dependencies:
    - requests: Library for sending HTTP requests

Note:
    To run the tests, make sure the ChemICR application server is running locally, and the server URL is correctly set
    in the `setup_client` function.

Author:
    Dr. Aleksei Krasnov
    a.krasnov@digital-science.com
    Date: December 4, 2023

"""

import base64
from pathlib import Path
from typing import List, Dict, Optional

from chemic.client import ChemClassifierClient
from chemic.config import Config

server_url=Config.API_URL

# Get the absolute path of the current file's directory
CURRENT_DIR = Path(__file__).resolve().parent

# Folder with images for tests
test_images = str(CURRENT_DIR / 'test_cases')


def perform_assertions(expected_result: Dict[str, str], actual_results: List[Dict[str, str]],image_id: Optional[str] = None):
    """
     Utility function to perform result assertions based on the provided test data.
    :param expected_result: dict
    :param actual_results: list
    """
    for result in actual_results:
        print('Actual result: ', result)
        predicted_value = result.get("predicted_label")
        current_image_id = image_id or result.get('image_id')
        print('current_image_id:', current_image_id)
        if not current_image_id:
            raise ValueError("Image ID could not be determined from the result.")

        assert expected_result[current_image_id] == predicted_value, (f"Prediction is wrong for {current_image_id}. "
                                             f"Expected {expected_result[current_image_id]}, got {predicted_value}.")
        # Reset image_id to None to handle multiple results correctly
        image_id = None


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
        base64_data = base64.b64encode(img_data)
    return base64_data


def classify(image_path=None, image_data=None):
    # Sets up and returns an instance of ChemRecognitionClient for testing.
    client = ChemClassifierClient(server_url=server_url)
    results = client.classify_image(image_path=image_path, image_data=image_data)
    print('Classification results: ', results)
    return results


def test_healthcheck():
    """
    Tests the health check functionality of the ChemICR application.
    """
    # Test health check
    client = ChemClassifierClient(server_url=server_url)
    health_status = client.healthcheck().get('status')
    assert health_status == "Server is up and running", f"Health check failed. Server response: {health_status}"


def test_classify_single_structure():
    """
    Tests the prediction of chemical structures for single structure images by Molscribe.
    """
    expected_result = {
        'EP-1678168-B1_image_102.tif': 'single chemical structure',
        'EP-1678168-B1_image_201.tif': 'single chemical structure',
        'EP-1678168-B1_image_497.tif': 'single chemical structure',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_497.tif'
    recognition_results = classify(image_path=image_path)
    assert recognition_results, f'Molscribe returned empty list {recognition_results}. Check Molscribe library'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results)



def test_classify_reaction():
    """
    Tests the prediction of chemical reactions for reaction images.
    """
    expected_result = {
        'EP-1678168-B1_image_385.tif': "chemical reactions"
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_385.tif'
    # Send the request and get the result
    recognition_results = classify(image_path=image_path)
    assert recognition_results, f'Rxnscribe returned empty list {recognition_results}. Check Rxnscribe library'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results)


def test_classify_no_structure():
    """
    Tests the scenario where there is no chemical structure in the image.
    """
    expected_result = {
        "cat_3.jpg": "no chemical structures",
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/cat_3.jpg'
    # Send the request and get the result
    recognition_results = classify(image_path=image_path)
    assert recognition_results, f'Empty list returned for image without chemical data'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,)


def test_classify_image_as_base64():
    """
    Tests the prediction of chemical structures for single structure base64 encoded image object by Molscribe.
    """
    expected_result = {
        'EP-1678168-B1_image_497.tif': "single chemical structure",
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_497.tif'
    base64_data = image_to_base64(image_path)
    recognition_results = classify(image_data=base64_data)
    assert recognition_results, (f'Molscribe returned empty list {recognition_results}. Check implementation of'
                                 f' Molscribe library in ChemICR')
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       image_id=Path(image_path).name
                       )


def test_classify_multiple_structure():
    """
    Tests the prediction of multiple chemical structures in an image.
    """
    expected_result = {
        'WO-2021090855-A1_image_1713.tif': 'multiple chemical structures',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/WO-2021090855-A1_image_1713.tif'
    # Send the request and get the result
    recognition_results = classify(image_path=image_path)
    assert recognition_results, (f'Decimer returns empty list {recognition_results}. '
                                 f'Check if Decimer and decimer-segmentation implementations.')
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,)

def test_prediction_images_from_dir():
    """
    Tests the prediction of several images from the provided path to dirctory.
    """
    expected_result = {
        '85.png': 'single chemical structure',
        'caffeine.png': 'single chemical structure',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/test_images_dir'
    # Send the request and get the result
    recognition_results = classify(image_path=image_path)
    assert recognition_results, f'ChemICR returns {recognition_results}'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
)
