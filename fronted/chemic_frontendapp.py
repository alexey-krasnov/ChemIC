import base64
import os
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image
from streamlit_navigation_bar import st_navbar

from about import show_about
from chemic.config import Config
from docs import show_docs

# Define your API URL
# API_URL = "http://127.0.0.1:5010"  # Update with your actual API endpoint if different
API_URL = Config.API_URL

def show_footer():
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
            padding: 5px;
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
    try:
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = requests.post(f"{API_URL}/classify_image", data={"image_data": img_base64})

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                result['image_id'] = image_file.name
                result['image_preview'] = img_base64  # Add base64 image string
                return result
            elif isinstance(result, list) and len(result) > 0:
                result[0]['image_id'] = image_file.name
                result[0]['image_preview'] = img_base64  # Add base64 image string
                return result[0]
            else:
                st.error("Unexpected response format.")
                return None
        else:
            st.error(f"Error: {response.json().get('error')}")
            return None
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

def classify_multiple_images(image_files):
    results = []
    for image_file in image_files:
        result = classify_image_from_file(image_file)
        if result:
            results.append(result)
    return results

def classify_image_from_path(image_path):
    try:
        response = requests.post(f"{API_URL}/classify_image", data={"image_path": image_path})

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                result['image_id'] = image_path
                return result
            elif isinstance(result, list) and len(result) > 0:
                return result
            else:
                st.error("Unexpected response format.")
                return None
        else:
            st.error(f"Error: {response.json().get('error')}")
            return None
    except Exception as e:
        st.error(f"Error processing the image path: {e}")
        return None

def create_csv_download_link(df):
    df.drop('image_preview', axis=1, inplace=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    csv = df.to_csv(index=True).encode('utf-8')  # Include index for CSV
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"classification_results_{timestamp}.csv",
        mime="text/csv",
    )

def show_home():
    st.title("Chemical Image Classifier")
    # st.write("Upload one or more images or provide an image path to classify their chemical content.")

    st.sidebar.header("Options")
    current_mode = st.sidebar.radio("Select Input Mode", ["Upload Images", "Input Image Path: Local Server Run"])

    # Initialize session state variables
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Update mode and clear results if mode changes
    if st.session_state.mode != current_mode:
        st.session_state.mode = current_mode
        st.session_state.results = None

    if current_mode == "Upload Images":
        st.write("Upload one or more images to classify their chemical content.")

        uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "tiff", "tif"], accept_multiple_files=True)
        if uploaded_files:
            results = classify_multiple_images(uploaded_files)
            if results:
                st.session_state.results = results

    elif current_mode == "Input Image Path: Local Server Run":
        st.write("Local Server Run Only: Provide an image path to classify their chemical content.")
        image_path = st.text_input("Enter the image path:")
        if st.button("Classify Images"):
            result = classify_image_from_path(image_path)
            if result:
                st.session_state.results = result

    if st.session_state.results:
        st.write("Classification Results:")

        # Generate image previews and CSV data
        classification_results = []

        for result in st.session_state.results:
            image_preview = f'<img src="data:image/png;base64,{result.get("image_preview")}" width="100"/>'
            classification_results.append({
                'image_id': result.get('image_id'),
                'image_preview': image_preview,  # Add image preview HTML
                'predicted_label': result.get('predicted_label', 'no chemical structures'),
                'classifier_package': result.get('classifier_package', 'ChemIC-ml_1.3'),
                'classifier_model': result.get('classifier_model', 'ResNet_50')
            })

        classification_df = pd.DataFrame(classification_results)

        html = classification_df.to_html(escape=False, index=True)  # Ensure HTML is not escaped
        st.markdown(html, unsafe_allow_html=True)  # Render HTML in Streamlit

        # Create CSV download link with index and discard image previews
        create_csv_download_link(df=classification_df)

    if st.sidebar.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/healthcheck")
            if response.status_code == 200:
                st.sidebar.success("API is up and running!")
            else:
                st.sidebar.error("API is not healthy.")
        except Exception as e:
            st.sidebar.error(f"Error connecting to API: {e}")

def main():
    st.set_page_config(
        page_title="Chemical Image Classifier",
        page_icon=":test_tube:",
        layout="wide",
    )

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(parent_dir, "public/ChemIC.svg")

    pages = [
        "Home",
        "About",
        "API Documentation",  # Add the new page
        "GitHub",
    ]

    urls = {"GitHub": "https://github.com/ontochem/ChemIC"}

    styles = {
        "nav": {
            "background-color": "#333333",  # Dark gray for the header bar
            "justify-content": "middle",
            "font-family": "Bahnschrift, sans-serif",
            "padding": "0 10px",  # Reduced padding for closer buttons
        },
        "img": {
            "padding-right": "2px",
            "background-color": "#FFFFFF",  # White background behind the logo for visibility
            "border-radius": "4px",  # Rounded edges for the logo background
        },
        "span": {
            "color": "#FFFFFF",  # White text color for the navbar buttons
            "padding": "10px",
            "font-family": "Bahnschrift, sans-serif",
            "background-color": "#555555",  # Medium gray background for buttons in normal state
            "margin": "0 5px",  # Spacing between the buttons
            "border-radius": "3px",  # Slight rounding for buttons
        },
        "active": {
            "color": "#FFFFFF",  # White text color for active buttons
            "background-color": "#777777",  # Light gray for active buttons
            "font-weight": "bold",  # Make the active button text bold
            "padding": "10px",
            "font-family": "Bahnschrift, sans-serif",
            "border-radius": "3px",  # Matching rounding for active buttons
        },
        "hover": {
            "background-color": "#444444",  # Slightly darker gray for buttons on hover
            "color": "#FFFFFF",  # White text color for hover state
        }
    }

    options = {
        "show_menu": False,
        "show_sidebar": False,
    }

    page = st_navbar(
        pages,
        logo_path=logo_path,
        urls=urls,
        styles=styles,
        options=options,
    )

    functions = {
        "Home": show_home,
        "About": show_about,
        "API Documentation": show_docs,  # Add the new page
    }

    go_to = functions.get(page)
    if go_to:
        go_to()

    show_footer()

if __name__ == "__main__":
    main()
