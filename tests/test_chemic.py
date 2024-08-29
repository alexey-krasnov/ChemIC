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

from typing import List, Dict, Optional
from chemic.client import ChemClassifierClient
from pathlib import Path
import time
import base64

server_url='http://127.0.0.1:5000'

# Get the absolute path of the current file's directory
CURRENT_DIR = Path(__file__).resolve().parent

# Folder with images for tests
test_images = str(CURRENT_DIR / 'test_cases')

# Adjust the time interval depending on time.sleep(1) in image_processing.process_images
TIME_TO_CLASSIFY = 5


def perform_assertions(expected_result: Dict[str, str], actual_results: List[Dict[str, str]], key_name: str, image_id: Optional[str] = None):
    """
     Utility function to perform result assertions based on the provided test data.
    :param expected_result: dict
    :param actual_results: list
    :param key_name: str
    """
    for result in actual_results:
        print('Actual result: ', result)
        predicted_value = result.get(key_name)
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


def classify_recognize(image_path=None, image_data=None, single_mol_recognizer=None, multiple_mols_recognizer=None):
    # Sets up and returns an instance of ChemRecognitionClient for testing.
    client = ChemRecognitionClient(server_url=server_url)
    client.server_empty_all_queues()
    client.classify_images(image_path=image_path, image_data=image_data)
    time.sleep(TIME_TO_CLASSIFY)
    client.recognize_images(single_mol_recognizer=single_mol_recognizer,
                            multiple_mols_recognizer=multiple_mols_recognizer,
                            use_chemicr_db=False) # Not use ChemICR database for this test
    recognition_results = client.retrieve_results()
    print('Recognition results: ', recognition_results)
    return recognition_results

#
# def test_healthcheck():
#     """
#     Tests the health check functionality of the ChemICR application.
#     """
#     # Test health check
#     client = ChemRecognitionClient(server_url=server_url)
#     health_status = client.healthcheck().get('status')
#     assert health_status == "Server is up and running", f"Health check failed. Server response: {health_status}"
#

def test_molscribe_prediction_single_structure():
    """
    Tests the prediction of chemical structures for single structure images by Molscribe.
    """
    expected_result = {
        # 'EP-1678168-B1_image_102.tif': 'CSc1nc(N)nc(N[C@@H](C)C(=O)Nc2cccc(N)c2)c1C#N',
        # 'EP-1678168-B1_image_201.tif': 'CCOc1nc(N)nc(N(C)[C@@H](C)C(=O)Nc2cccc(C(F)(F)F)c2)c1C#N',
        'EP-1678168-B1_image_497.tif': 'CCOc1nc(N)nc(N[C@@H](CCCN2CCOCC2)C(=O)Nc2cccc(C(F)(F)F)c2)c1C#N',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_497.tif'
    recognition_results = classify_recognize(image_path=image_path, single_mol_recognizer='molscribe')
    assert recognition_results, f'Molscribe returned empty list {recognition_results}. Check Molscribe library'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')


def test_decimer_prediction_single_structure():
    """
    Tests the prediction of chemical structures for single structure images by DECIMER.
    """
    expected_result = {
        'EP-1678168-B1_image_109.tif': 'C[C@@H](C(=O)NC1=CC(=CC(=C1)C(=O)NCCN(C)C)C(F)(F)F)NC2=NC(=NC(=C2C#N)SC)N',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_109.tif'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path, single_mol_recognizer='decimer')
    assert recognition_results, f'Decimer returned empty list {recognition_results}. Check Molscribe library'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')


def test_rxnscribe_prediction_reaction():
    """
    Tests the prediction of chemical reactions for reaction images.
    """
    expected_result = {
        'EP-1678168-B1_image_385.tif': "O=C(O)COCc1ccccc1>>O=C(COCc1ccccc1)Nc1cccc(C(F)(F)F)c1conditions:Nc1cccc(C(F)(F)F)c1"
                                       "\nO=C(COCc1ccccc1)Nc1cccc(C(F)(F)F)c1>>O=C(CO)Nc1cccc(C(F)(F)F)c1"
                                       "\nO=C(CO)Nc1cccc(C(F)(F)F)c1>>O=C(COc1nc(Cl)nc(Cl)n1)Nc1cccc(C(F)(F)F)c1conditions:Clc1nc(Cl)nc(Cl)n1"
                                       "\nO=C(COc1nc(Cl)nc(Cl)n1)Nc1cccc(C(F)(F)F)c1>>O=C(O)COCc1ccccc1",
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_385.tif'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path)
    assert recognition_results, f'Rxnscribe returned empty list {recognition_results}. Check Rxnscribe library'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='rsmi')


def test_image_prediction_no_structure():
    """
    Tests the scenario where there is no chemical structure in the image.
    """
    expected_result = {
        'cat_3.jpg': None,
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/cat_3.jpg'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path)
    assert recognition_results, f'Empty list returned for image without chemical data'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')


def test_molscribe_prediction_imgage_as_base64():
    """
    Tests the prediction of chemical structures for single structure base64 encoded image object by Molscribe.
    """
    expected_result = {
        'EP-1678168-B1_image_497.tif': 'CCOc1nc(N)nc(N[C@@H](CCCN2CCOCC2)C(=O)Nc2cccc(C(F)(F)F)c2)c1C#N',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_497.tif'
    base64_data = image_to_base64(image_path)
    recognition_results = classify_recognize(image_data=base64_data, single_mol_recognizer='molscribe')
    assert recognition_results, (f'Molscribe returned empty list {recognition_results}. Check implementation of'
                                 f' Molscribe library in ChemICR')
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles',
                       image_id=Path(image_path).name
                       )


def test_decimer_prediction_imgage_as_base64():
    """
    Tests the prediction of chemical structures for single structure base64 encoded image object by Decimer.
    """
    expected_result = {
        'EP-1678168-B1_image_497.tif': 'CCOC1=C(C#N)C(=NC(=N1)N)N[C@@H](CCCN2CCOCC2)C(=O)NC3=CC=CC(=C3)C(F)(F)F',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/EP-1678168-B1_image_497.tif'
    base64_data = image_to_base64(image_path)
    recognition_results = classify_recognize(image_data=base64_data, single_mol_recognizer='decimer')
    assert recognition_results, (f'Decimer recognition results: {recognition_results}. Check implementation of'
                                 f' Decimer library in ChemICR')
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles',
                       image_id=Path(image_path).name
                       )

# #FIXME this fails  because decimer segmentaton cannot handle terminated atoms on image US-20220048929-A1_image_1674.TIF
# def test_decimer_prediction_multiple_structure_free_fromat():
#     """
#     Tests the prediction of multiple chemical structures in an image.
#     """
#     expected_result = {
#             'US-20220048929-A1_image_1674_0.TIF': 'C=C(C1=C(C=C(C(=C1)O)C(=O)O)O)O',
#             'US-20220048929-A1_image_1674_1.TIF': 'C=C(C1=CC2=CC(=C(C=C2C=C1O)C(=C)O)O)O',
#             'US-20220048929-A1_image_1674_2.TIF': 'C1=C(C=C(C(=C1)O)C(=O)O)[R17a]([H])([H])C2=CC(=C(C=C2)O)C(=O)O',
#             'US-20220048929-A1_image_1674_3.TIF': 'CC1=C(C=CC(=C1)[R19a]([H])([H])C2=CC=C(C(=C2)O)C(=O)O)C(=O)C',
#     }
#     # Absolute path to the image file or directory you want to classify
#     image_path = f'{test_images}/US-20220048929-A1_image_1674.TIF'
#     # Send the request and get the result
#     recognition_results = classify_recognize(image_path=image_path,  multiple_mols_recognizer='decimer')
#     assert recognition_results, (f'Decimer returns empty list {recognition_results}. '
#                                  f'Check if Decimer and decimer-segmentation implementations.')
#     perform_assertions(expected_result=expected_result,
#                        actual_results=recognition_results,
#                        key_name='smiles')


def test_decimer_prediction_multiple_structure():
    """
    Tests the prediction of multiple chemical structures in an image.
    """
    expected_result = {
        'WO-2021090855-A1_image_1713_0.tif': 'CS(=O)(=O)CC[C@@H](C(=O)O)NC(=O)OCC1C2=C(C=CC=C2)C3=C1C=CC=C3',
        'WO-2021090855-A1_image_1713_1.tif': 'C=C(C)C[C@@H](C(=O)O)NC(=O)OCC1C2=C(C=CC=C2)C3=C1C=CC=C3',
        'WO-2021090855-A1_image_1713_2.tif': 'C[C@H]([C@@H](C(=O)O)N(C)C(=O)OCC1C2=C(C=CC=C2)C3=C1C=CC=C3)O',
        'WO-2021090855-A1_image_1713_3.tif': 'C[C@H]([C@@H](C(=O)O)N(C)C(=O)OCC1C2=C(C=CC=C2)C3=C1C=CC=C3)OCC4=CC=CC=C4',
        'WO-2021090855-A1_image_1713_4.tif': 'CN([C@@H](CC1=CC=C(C=C1)OC)C(=O)O)C(=O)OCC2C3=C(C=CC=C3)C4=C2C=CC=C4',
        'WO-2021090855-A1_image_1713_5.tif': 'CC(C)[C@@H](C(=O)O)N(C)C(=O)OCC1C2=C(C=CC=C2)C3=C1C=CC=C3',
        'WO-2021090855-A1_image_1713_6.tif': 'C1=CC2=C(C=C1)C(COC(=O)N3CCOC[C@H]3C(=O)O)C4=C2C=CC=C4',

    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/WO-2021090855-A1_image_1713.tif'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path,  multiple_mols_recognizer='decimer')
    assert recognition_results, (f'Decimer returns empty list {recognition_results}. '
                                 f'Check if Decimer and decimer-segmentation implementations.')
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')


def test_osra_prediction_multiple_structure():
    """
    Tests the prediction of multiple chemical structures in an image.
    """
    expected_result = {
        'US-08680111-B2_image_802.TIF': 'C1NCc2ccccc12'
                                        '\nC1NSc2ccccc12'
                                        '\nC1c2ccccc2SC1'
                                        '\nO=S1(c2ccccc2CC1)=O'
                                        '\nO=S1(NCc2ccccc12)=O'
                                        '\nC1Cc2ccccc2C1',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/US-08680111-B2_image_802.TIF'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path, multiple_mols_recognizer='osra')
    assert recognition_results, f'OSRA returns {recognition_results}. Check if OSRA web service is running.'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')


def test_prediction_images_from_dir():
    """
    Tests the prediction of several images from the provided path to dirctory.
    """
    expected_result = {
        '85.png': 'O=C/C=C/c1ccccc1',
        'caffeine.png': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    }
    # Absolute path to the image file or directory you want to classify
    image_path = f'{test_images}/test_images_dir'
    # Send the request and get the result
    recognition_results = classify_recognize(image_path=image_path,  single_mol_recognizer='molscribe')
    assert recognition_results, f'ChemICR returns {recognition_results}'
    perform_assertions(expected_result=expected_result,
                       actual_results=recognition_results,
                       key_name='smiles')
