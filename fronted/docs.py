import streamlit as st


def show_docs():
    st.title("ChemIC API Documentation")

    st.write("""
    ## Overview

    The ChemIC API provides endpoints for classifying chemical images. This documentation covers the available endpoints, request formats, and example `curl` commands.

    ## API Endpoints

    ### 1. Classify Image

    **Endpoint**: `/classify_image`  
    **Method**: `POST`  
    **Description**: Classifies an image based on either the image path or image data.

    **Request Parameters**:
    - `image_path` (optional): Path to the image file on the server.
    - `image_data` (optional): Base64-encoded image data.

    **Responses**:
    - Returns a list of classification results in JSON format.

    **Example Request**:
    
    **Using image data:**
    ```bash
    curl -X POST "http://127.0.0.1:5010/classify_image" \
         -F "image_data=<base64_image_data>"
    ```

    **Using image path:**
    ```bash
    curl -X POST "http://127.0.0.1:5010/classify_image" \
         -F "image_path=<image_path>"
    ```

    ### 2. Health Check

    **Endpoint**: `/healthcheck`  
    **Method**: `GET`  
    **Description**: Checks the health status of the API.

    **Example Request:**
    ```bash
    curl -X GET "http://127.0.0.1:5010/healthcheck"
    ```

    ## Response Format

    **Classify Image**:
    - Returns a list of objects with the following fields:
      - `image_id`: The identifier of the image.
      - `predicted_label`: The label predicted by the classifier.
      - `classifier_package`: The package used for classification.
      - `classifier_model`: The model used for classification.

    **Health Check**:
    - Returns a JSON object with a `status` field indicating the server's health.

    ## Notes

    - Ensure that your FastAPI server is running before making requests.
    - For image data, make sure the base64 string is properly encoded.
    """)

