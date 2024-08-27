# Chemical Image Classifier (ChemIC) v1.2
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/ontochem/ChemIC/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/ontochem/ChemIC.svg)](https://github.com/ontochem/ChemIC/issues)
[![GitHub contributors](https://img.shields.io/github/contributors/ontochem/ChemIC.svg)](https://github.com/ontochem/ChemIC/graphs/contributors)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10546827.svg)](https://doi.org/10.5281/zenodo.10546827)

## Table of Contents
- [Project Description](#project-description)
- [Requirements](#requirements)
- [Prepare Workspace Environment with Conda](#prepare-workspace-environment-with-conda)
- [Model construction](#model-construction)
- [Models](#models)
- [Usage Web Service for Chemical Image Classification](#usage-web-service-for-chemical-image-classification)
- [Jupyter Notebook](#jupyter-notebook)
- [Author](#author)
- [Citation](#citation)
- [References](#references)
- [License](#license)

## Project Description
The Chemical Image Classifier (ChemIC) project can be suitable for training and using
a CNN model for classification chemical images into one of the four predefined classes:
1. images with single chemical structure;
2. images with chemical reactions; 
3. images multiple chemical structures; 
4. images with no chemical structures.


The package consists of three main components:
### A) Implementation of Creating of Convolutional Neural Network (CNN) model for Image Classification ([chemic_train_eval.py](chemic_train_eval.py)):
- Responsible for training a deep learning model to classify images into four predefined classes.
- Uses a pre-trained ResNet-50 model and includes data preparation, model training, evaluation, and testing steps.

### B) Web Service for Chemical Image Classification ([app.py](chemic/app.py)):
- Provides a Flask web application for classifying chemical images using the trained ResNet-50 model.
- Exposes an endpoint /classify_image for accepting chemical images and returning the predicted class.

### C) Image Classification Client ([client.py](chemic/client.py)):
- Interact with the ChemIC web-server. The client can send to server:
  - the path to a individual image file
  - the path to directory with several images
  - base64 encoded image data object,

  and the server classifies the images, providing the client with the recognition results.

## Requirements
* Flask>=3.0.0
* gunicorn>=21.2.0
* numpy>=1.26.3
* pandas>=2.2.0
* pillow>=10.2.0
* requests>=2.31.0
* scikit-learn>=1.3.2
* torch>=2.2.0
* torchmetrics>=1.2.1
* torchvision>=0.17.0

## Prepare Workspace Environment with Conda
```bash
# 1. Create and activate conda environment
conda create --name chemic "python<3.12"
conda activate chemic

# 2. Install ChemIC-ml
# 2.1 from PyPi
pip install ChemIC-ml

# 2.2 Or get and install package from GitHub repository
pip install git+https://github.com/ontochem/ChemIC.git

# 2.3 Or install in the editable mode from GitHub repository
git clone https://github.com/ontochem/ChemIC.git
cd ChemIC
pip install -r requirements.txt
pip install -e .
```
- Where -e means "editable" mode.
 
## Model construction
First download the archive with manually labeled images as a part of Supplementary materials from Zenodo [dataset_for_image_classifier.zip](https://zenodo.org/records/13378718).
Unzip the archive with images. 
```bash
unzip dataset_for_image_classifier.zip
```
To perform model training, validation, and test steps as well as saving your own trained model run in CLI:
```bash
python chemic_train_eval.py --dataset_dir /path/to/data --checkpoint_path /path/to/checkpoint.pth --models_dir /path/to/models
```
- **--dataset_dir** is the directory containing the dataset (train, test, validation subdirectories).
- **--checkpoint_path** is the path to the existing model checkpoint file.
- **--models_dir** is the directory to save new trained models. 

This will execute the training and evaluation using the specified paths.

## Models download
Download already pretrained models from Zenodo as archive [models.zip](https://doi.org/10.5281/zenodo.10709886) and unzip it to the directory `chemic/models`.
The directory `models` should contain the pretrained model `chemical_image_classifier_resnet50.pth` for chemical image classification.

## Usage Web Service for Chemical Image Classification

### 1. Start the Flask web server in a production mode
Run in command line from the directory ChemIC:
```bash
gunicorn -w 1 -b 127.0.0.1:5000 --timeout 3600 chemic.app:app
```
- -w 1: Specifies the number of worker processes. In this case, only one worker is used.
  Adjust this value based on your server's capabilities.
- -b 127.0.0.1:5000: Binds the application to the specified address and port. Change
  the address and port as needed.
- --timeout 3600: Sets the maximum allowed request processing time in seconds.
  Adjust this value based on your application's needs.

## 2. Classify Image with client.py module in CLI
```bash
 python chemic/client.py --image_path /path/to/images --export_dir /path/to/export
```
OR 
```bash
 python chemic/client.py  --image_data <base64_encoded_string> --export_dir /path/to/export
```
- **--image_path** is the path to the image file or directory with images for classification.
- **--image_data** is the base64 encoded image data.
- **--export_dir** is the export directory for the results.

## 3. Or use client for classification in your Python code
```python
from chemic.client import ChemClassifierClient

client = ChemClassifierClient(server_url='http://127.0.0.1:5000')
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
[{'image_id': 'image_name_1.png',
  'predicted_label': 'single chemical structure',
  'program': 'ChemIC',
  'program_version': '1.2'},
 {'image_id': 'image_name_2.png',
  'predicted_label': 'multiple chemical structures',
  'program': 'ChemIC',
  'program_version': '1.2'},
  ...
]
```
## Jupyter Notebook
The `client_image_classifier.ipynb` Jupyter notebook in folder `notebooks` provides an easy-to-use interface for classifying images.
Follow the outlined steps to perform image classification.

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
This project is licensed under the MIT - see the LICENSE.md file for details.