# Chemical Image Classifier (ChemIC) v1.3.1
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/ontochem/ChemIC/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/ontochem/ChemIC.svg)](https://github.com/ontochem/ChemIC/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/ontochem/ChemIC.svg)](https://github.com/ontochem/ChemIC/graphs/contributors)
[![DOI](https://zenodo.org/badge/DOI/10.1039/D3DD00228D.svg)](https://doi.org/10.1039/D3DD00228D)



## Table of Contents
- [Project Description](#project-description)
- [Requirements](#requirements)
- [Prepare Workspace Environment with Conda](#prepare-workspace-environment-with-conda)
- [Model Construction](#model-construction)
- [Models Download](#models-download)
- [Usage: Web Service for Chemical Image Classification](#usage-web-service-for-chemical-image-classification)
- [Jupyter Notebook](#jupyter-notebook)
- [Author](#author)
- [Citation](#citation)
- [References](#references)
- [License](#license)

## Project Description
The Chemical Image Classifier (ChemIC) project is designed for training and utilizing a Convolutional Neural Network (CNN) model to classify chemical images into one of four predefined classes:
1. Images with a single chemical structure
2. Images with chemical reactions
3. Images with multiple chemical structures
4. Images with no chemical structures

The package consists of three main components:

### A) CNN Model for Image Classification ([chemic_train_eval.py](chemic_train_eval.py))
- Trains a deep learning model to classify images into the four predefined classes.
- Utilizes a pre-trained ResNet-50 model and includes steps for data preparation, model training, evaluation, and testing.

### B) Web Service for Chemical Image Classification ([app.py](chemic/app.py))
- Provides a FastAPI web application for classifying chemical images using the trained ResNet-50 model.
- Exposes an endpoint `/classify_image` for accepting chemical images and returning the predicted class.

### C) Image Classification Client ([client.py](chemic/client.py))
- Interacts with the ChemIC web server. The client can send to the server:
  - The path to an individual image file
  - The path to a directory with multiple images
  - Base64 encoded image data

  The server classifies the images and returns the recognition results to the client.

## Prepare Workspace Environment with Conda
```bash
# 1. Create and activate the conda environment
conda create --name chemic "python<3.12"
conda activate chemic

# 2. Install ChemIC-ml
# 2.1 From PyPI
pip install ChemIC-ml

# 2.2 Or, install from the GitHub repository
pip install git+https://github.com/ontochem/ChemIC.git

# 2.3 Or, install in editable mode from the GitHub repository
git clone https://github.com/ontochem/ChemIC.git
cd ChemIC
pip install -r requirements.txt
pip install -e .
```
- Where -e means "editable" mode.
 
## Model construction
First, download the archive with manually labeled images, available as part of the supplementary materials from Zenodo: [dataset_for_image_classifier.zip](https://zenodo.org/records/13378718). Unzip the archive:
```bash
unzip dataset_for_image_classifier.zip
```
To perform model training, validation, and testing, as well as saving your trained model, run the following command in the CLI:
```bash
python chemic_train_eval.py --dataset_dir /path/to/data --checkpoint_path /path/to/checkpoint.pth --models_dir /path/to/models
```
* `--dataset_dir`: Directory containing the dataset (with train, test, and validation subdirectories).
* `--checkpoint_path`: Path to the existing model checkpoint file.
* `--models_dir`: Directory to save newly trained models.

This command executes the training and evaluation using the specified paths.

## Models download
Download the pre-trained models from Zenodo as an archive: [models.zip](https://doi.org/10.5281/zenodo.10709886). 
Unzip it into the `chemic/models` directory. The models directory should contain the pre-trained model `chemical_image_classifier_resnet50.pth` for chemical image classification.

## Usage Web Service for Chemical Image Classification
### 1. Start the FastAPI Web Server in Production Mode
Run the following command in terminal:
```bash
uvicorn chemic.app:app --host 127.0.0.1 --port 5010 --workers 1 --timeout-keep-alive 3600
```
* `--workers 1`: Specifies the number of worker processes. Adjust based on your server's capabilities.
* `--host 127.0.0.1 --port 5010`: Binds the application to the specified address and port. Modify as needed.
* `--timeout-keep-alive 3600`: Sets the maximum allowed request processing time in seconds. Adjust as necessary.

## 2. Use frontend Web interface
In another terminal window, run the following command:
```bash
streamlit run chemic_frontendapp.py --server.address=0.0.0.0 --server.port=5009
```
This command will launch the ChemIC user web interface.

## 3. Classify Images Using the `client.py` Module via CLI
```bash
 python chemic/client.py --image_path /path/to/images --export_dir /path/to/export
```
OR 
```bash
 python chemic/client.py  --image_data <base64_encoded_string> --export_dir /path/to/export
```
* `--image_path` is the path to the image file or directory with images for classification.
* `--image_data` is the base64 encoded image data.
* `--export_dir` is the export directory for the results.

## 4. Alternatively, Use the Client for Classification in Your Python Code
```python
from chemic.client import ChemClassifierClient

client = ChemClassifierClient(server_url='http://127.0.0.1:5010')

# Check the health of the server
health_status = client.healthcheck().get('status')
print(f"Health Status: {health_status}")

# Use image path or directory. Replace with the actual path to your image file
image_path = '<path to the image file or directory with images for classification>'
recognition_results = client.classify_image(image_path)

# OR use base64-encoded image data. Replace with your base64-encoded image data:
base64_data = b'iVBORw0KGgoAAAANSUhEUgA....'
recognition_results = client.classify_image(image_data=base64_data)

# Recognition results will be returned in the form of  a list of dictionaries
print(recognition_results)
[
  {
    'image_id': 'image_name_1.png',
    'predicted_label': 'single chemical structure',
    'classifier_package': 'ChemIC-ml_1.3.1',
    'classifier_model': 'ResNet_50',
  },
 {
   'image_id': 'image_name_2.png',
    'predicted_label': 'multiple chemical structures',
   'classifier_package': 'ChemIC-ml_1.3.1',
   'classifier_model': 'ResNet_50',
 },
  ...
]
```

## Jupyter Notebook
The [client_image_classifier.ipynb](notebooks/client_image_classifier.ipynb) notebook in the `notebooks` directory provides an easy-to-use interface for classifying images. Follow the steps outlined in the notebook to perform image classification.

## Author
Dr. Aleksei Krasnov
a.krasnov@digital-science.com
OntoChem GmbH part of Digital Science

## Citation
- A. Krasnov, S. Barnabas, T. Böhme, S. Boyer, L. Weber, Comparing software tools for optical chemical structure recognition, Digital Discovery (2024).	https://doi.org/10.1039/D3DD00228D
- L. Weber, A. Krasnov, S. Barnabas, T. Böhme, S. Boyer, Comparing Optical Chemical Structure Recognition Tools, ChemRxiv. (2023). https://doi.org/10.26434/chemrxiv-2023-d6kmg-v2

## References
- A. Krasnov, Images dataset for Chemical Images Classifier model. https://zenodo.org/records/13378718
- A. Krasnov, Chemical Image Classifier Model. https://zenodo.org/records/10709886

## License
This project is licensed under the MIT - see the [LICENSE.md](LICENSE.md) file for details.