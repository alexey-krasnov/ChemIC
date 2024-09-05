import streamlit as st

def show_docs():
    st.title("ChemIC API Documentation")

    st.write("""
    ## Overview

    The ChemIC API provides endpoints for classifying chemical images. This documentation outlines the available endpoints, request formats, and example `curl` commands.
    
    ## API Base URL

    **Base URL**: `http://SERVER:PORT`
    
    - Replace `SERVER` with the server IP address or domain.
    - Replace `PORT` with the port on which the API is running.
    - Example default local setup: `http://127.0.0.1:5010`

    ## API Endpoints

    ### 1. Classify Image

    **Endpoint**: `/classify_images`  
    **Method**: `POST`  
    **Description**: Classifies an image based on either an image file path or image data.

    **Request Parameters**:
    - `image_path` (optional): Path to the image file located on the server.
    - `image_data` (optional): Base64-encoded image data for direct submission.

    **Response**:
    - A JSON list of classification results.

    **Example Request**:

    **Using base64-encoded image data:**
    ```bash
    curl -X POST "http://127.0.0.1:5010/classify_images" -F "image_data=<base64_image_data>"
    ```

    **Using image file path on the server:**
    ```bash
    curl -X POST "http://127.0.0.1:5010/classify_images" -F "image_path=<image_path>"
    ```

    ### 2. Health Check

    **Endpoint**: `/healthcheck`  
    **Method**: `GET`  
    **Description**: Retrieves the health status of the ChemIC API.

    **Example Request**:
    ```bash
    curl -X GET "http://127.0.0.1:5010/healthcheck"
    ```

    ## Response Format

    **Classify Image Response**:
    The API returns a list of objects in JSON format, each with the following fields:
    - `image_id`: Identifier for the image.
    - `predicted_label`: Label predicted by the image classifier.
    - `classifier_package`: The classification package used.
    - `classifier_model`: The machine learning model applied for classification.

    Example:
    ```json
    [
        {
            "image_id": "image_name_1.png",
            "predicted_label": "single chemical structure",
            "classifier_package": "ChemIC-ml_1.3.1",
            "classifier_model": "ResNet_50"
        },
        {
            "image_id": "image_name_2.png",
            "predicted_label": "multiple chemical structures",
            "classifier_package": "ChemIC-ml_1.3.1",
            "classifier_model": "ResNet_50"
        }
    ]
    ```

    **Health Check Response**:
    - Returns a JSON object with a `status` field indicating the server's health.
    ```json
    {
        "status": "Server is up and running"
    }
    ```

    ## Notes

    - Ensure that the FastAPI server is running and accessible before making API requests.
    - For base64-encoded images, verify that the encoding is correct to avoid errors in classification.
    """)