"""
ChemIC Frontend using Streamlit.

Usage:
    streamlit run chemic_frontendapp.py --server.address=0.0.0.0 --server.port=8501
"""
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Define your API URL
API_URL = "http://127.0.0.1:5000"  # Update with your actual API endpoint if different

def show_footer():
    """
    Display a fixed footer at the bottom of the Streamlit app.
    """
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f2f2f2;
            color: black;
            text-align: center;
            padding: 10px;
            z-index: 999;
        }
        </style>
        <div class="footer">
            <p>&copy; 2024 Chemical Image Classifier. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def classify_image_from_file(image_file):
    """
    Send an image file to the backend for classification.
    """
    try:
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send the image data to the API
        response = requests.post(f"{API_URL}/classify_image", data={"image_data": img_str})

        if response.status_code == 200:
            st.success("Image classified successfully!")
            return response.json()
        else:
            st.error(f"Error: {response.json().get('error')}")
            return None
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

def classify_image_from_path(image_path):
    """
    Send an image path to the backend for classification.
    """
    try:
        response = requests.post(f"{API_URL}/classify_image", data={"image_path": image_path})

        if response.status_code == 200:
            st.success("Image classified successfully!")
            return response.json()
        else:
            st.error(f"Error: {response.json().get('error')}")
            return None
    except Exception as e:
        st.error(f"Error processing the image path: {e}")
        return None

def main():
    """
    Main function for running the Streamlit app.
    """
    st.set_page_config(
        page_title="Chemical Image Classifier",
        page_icon=":test_tube:",
        layout="wide",
    )

    st.title("Chemical Image Classifier")
    st.write("Upload an image or provide an image path to classify its chemical content.")

    st.sidebar.header("Options")
    mode = st.sidebar.radio("Select Input Mode", ["Upload Image", "Input Image Path"])

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            results = classify_image_from_file(uploaded_file)
            if results:
                st.write("Classification Results:")
                st.json(results)

    elif mode == "Input Image Path":
        image_path = st.text_input("Enter the image path:")
        if st.button("Classify Image"):
            results = classify_image_from_path(image_path)
            if results:
                st.write("Classification Results:")
                st.json(results)

    # Health Check
    if st.sidebar.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/healthcheck")
            if response.status_code == 200:
                st.sidebar.success("API is up and running!")
            else:
                st.sidebar.error("API is not healthy.")
        except Exception as e:
            st.sidebar.error(f"Error connecting to API: {e}")

    show_footer()

if __name__ == "__main__":
    main()
