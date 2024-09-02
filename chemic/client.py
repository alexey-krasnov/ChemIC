import argparse
import datetime
import time
from pathlib import Path

import pandas as pd
import requests

from chemic.config import Config

start = time.time()

class ChemClassifierClient:
    """
    A client for interacting with the ChemIC application's server.

    Attributes:
        server_url (str): The URL of the ChemIC server.

    Methods:
        classify_image(image_path: str = None, image_data: bytes = None) -> dict:
            Sends a POST request to the server with either the specified image path or image data.
            Returns the classification results in dictionary format.

        healthcheck() -> str:
            Sends a GET request to the server to check its health status.
            Returns the health status as a string.
    """

    def __init__(self, server_url):
        """
        Initializes a ChemClassifierClient instance.

        Parameters:
            server_url (str): The URL of the ChemIC server.
        """
        self.server_url = server_url

    def classify_image(self, image_path: str = None, image_data: str = None):
        """
        Sends a POST request to the server with either the specified image path or image data.
        Returns the classification results in dictionary format.

        Parameters:
            image_path (str): The path to the image file.
            image_data (str): The base64 encoded data of the image.

        Returns:
            dict: Classification results, including image ID, predicted labels, and chemical structures.
        """
        try:
            data = {}
            if image_path:
                # Check if the path to the file is valid
                file_path = Path(image_path)
                if not file_path.exists():
                    raise FileNotFoundError(f'Invalid path provided: {image_path}')
                data['image_path'] = str(file_path)
            elif image_data:
                # Image data should be base64 encoded
                data['image_data'] = image_data
            else:
                raise ValueError("Either --image_path or --image_data must be provided.")

            # Send a POST request to the server
            response = requests.post(f'{self.server_url}/classify_image', data=data)
            # response.raise_for_status()  # Raise an HTTPError for bad responses

            print(response.json())
            # Parse the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {'error': str(e)}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'error': str(e)}

    def healthcheck(self):
        """
        Sends a GET request to the server to check its health status.
        Returns the health status as a string.

        Returns:
            str: The health status of the ChemIC server.
        """
        try:
            # Send a GET request to the server
            response = requests.get(f'{self.server_url}/healthcheck')
            response.raise_for_status()  # Raise an HTTPError for bad responses
            # Parse the JSON response
            return response.json()

        except requests.exceptions.RequestException as e:
            return {'error': str(e)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client for Chemical Images Classifier and Recognizer")
    parser.add_argument("--image_path", type=str, help="Path to the image file or directory with images.")
    parser.add_argument("--image_data", type=str, help="Base64 encoded image data.")
    parser.add_argument("--export_dir", type=str, default=".", help="Export directory for the results.")
    args = parser.parse_args()

    # URL for the combined image processing endpoint
    server_url = Config.API_URL

    # Create an instance of the client
    client = ChemClassifierClient(server_url)

    # Check the health of the server
    health_status = client.healthcheck().get('status')
    print(f"Health Status: {health_status}")

    # Send POST request to classify and predict images
    recognition_results = client.classify_image(image_path=args.image_path, image_data=args.image_data)

    print(recognition_results)

    # Convert results to pandas DataFrame
    df = pd.DataFrame(recognition_results)
    df.sort_values(by='image_id', inplace=True, ignore_index=True)
    print(df)
    df.to_csv(f'{args.export_dir}/{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}_classification_results.csv', index=False)

    end = time.time()
    print(f"Work took {time.strftime('%H:%M:%S', time.gmtime(end - start))}")
