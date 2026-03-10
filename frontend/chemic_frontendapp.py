import base64
import os
from datetime import datetime
from io import BytesIO, IOBase
import subprocess

import duckdb
import pandas as pd
import requests
import psutil
import streamlit as st
from PIL import Image
from pathlib import Path
from streamlit_navigation_bar import st_navbar

from about import show_about
from chemic.config import API_URL
from chemic.utils import encode_image_to_base64, generate_unique_identifier
from docs import show_docs

# Define your API URL
# API_URL = 'http://127.0.0.1:5010' # Adapt this URL to your own setup

MAX_UPLOAD_IMAGES = 100


# def save_to_duckdb(df):
#     """Save feedback DataFrame to DuckDB database."""
#     conn = duckdb.connect('chemic.db')
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS feedback (
#             image_id TEXT,
#             predicted_label TEXT,
#             corrected_label TEXT,
#             image_base64 TEXT
#         )
#     """)
#
#     # Insert rows one by one
#     for index, row in df.iterrows():
#         conn.execute("""
#             INSERT INTO feedback (image_id, predicted_label, corrected_label, image_base64)
#             VALUES (?, ?, ?, ?)
#         """, (row['image_id'], row['predicted_label'], row['corrected_label'], row['image_base64']))
#     print("Feedback saved to DuckDB")
#     conn.close()

def ensure_table_exists(conn):
    """
    Ensure the feedback table exists in the DuckDB database.

    Parameters:
    - conn: The DuckDB connection object.
    """
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            image_hash_id TEXT,
            predicted_label TEXT,
            corrected_label TEXT,
            image_base64 TEXT,
            PRIMARY KEY (image_hash_id, corrected_label)
        )
    ''')

def save_to_duckdb(df):
    """Save feedback DataFrame to DuckDB database, ensuring unique records."""
    conn = duckdb.connect('chemic.db')
    ensure_table_exists(conn)

    # Insert rows one by one
    for _, row in df.iterrows():
        image_base64 = row['image_base64']
        image_hash_id = generate_unique_identifier(base64_encoded_image=image_base64)

        # Insert only if the image_hash_id and corrected_label combination is not in the table
        conn.execute('''
            INSERT OR IGNORE INTO feedback (image_hash_id, predicted_label, corrected_label, image_base64)
            VALUES (?, ?, ?, ?)
        ''', (image_hash_id, row['predicted_label'], row['corrected_label'], row['image_base64']))

        # If the corrected label is different, we create a new record with the same image_hash_id
        if row['predicted_label'] != row['corrected_label']:
            conn.execute('''
                INSERT OR REPLACE INTO feedback (image_hash_id, predicted_label, corrected_label, image_base64)
                VALUES (?, ?, ?, ?)
            ''', (image_hash_id, row['predicted_label'], row['corrected_label'], row['image_base64']))

    print("Feedback saved to DuckDB")
    conn.close()

def is_uvicorn_running():
    """
    Checks if there's any running process with 'uvicorn' and 'chemic.app:app'.
    Returns True if such a process is found, otherwise False.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and "uvicorn" in cmdline and "chemic.app:app" in cmdline:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_uvicorn():
    log_file = open("uvicorn_log.txt", "w")
    process = subprocess.Popen(
        ["uvicorn", "chemic.app:app", "--host", "127.0.0.1", "--port", "5010", "--workers", "1", "--timeout-keep-alive", "3600"],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return process

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
            <p>&copy; 2024 ChemIC. Chemical Image Classifier. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def load_example_images():
    """
    Load example image paths from the example_images directory.
    """
    image_dir = Path("example_images").resolve()  # Ensure absolute path
    images = []

    # Iterate over all image files in the directory
    for image_path in image_dir.glob("*"):
        if image_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tiff', '.tif'):
            images.append(image_path)
    return images


def classify_image_from_file(image_file):
    try:
        # Ensure image_file is a file-like object
        if isinstance(image_file, (BytesIO, IOBase)):
            img = Image.open(image_file)
        else:
            # If image_file is a path, open it
            img = Image.open(image_file)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = requests.post(f"{API_URL}/classify_images", data={"image_data": img_base64})

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                result['image_id'] = getattr(image_file, 'name', 'unknown')
                result['image_preview'] = img_base64  # Add base64 image string
                return result
            elif isinstance(result, list) and len(result) > 0:
                result[0]['image_id'] = getattr(image_file, 'name', 'unknown')
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
    for image_file in image_files[:MAX_UPLOAD_IMAGES]:
        result = classify_image_from_file(image_file)
        if result:
            results.append(result)
    return results

def classify_image_from_path(image_path):
    try:
        response = requests.post(f"{API_URL}/classify_images", data={"image_path": image_path})

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
    if 'image_preview' in df.columns:
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
    st.title("Chemic: Chemical Image Classifier")

    st.sidebar.header("Options")
    current_mode = st.sidebar.radio("Select Input Mode", ["Upload Images", "Input Image Path: Local Server Run"])

    # Initialize session state variables
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Update mode and clear results if mode changes
    if st.session_state.mode != current_mode:
        st.session_state.mode = current_mode
        st.session_state.results = None
        st.session_state.uploaded_files = []
        st.rerun()

    # Load example images
    example_images = load_example_images()

    selected_example_images = st.sidebar.multiselect(
        "Choose example images",
        [img.name for img in example_images],
    )

    if current_mode == "Upload Images":
        st.write("Upload one or more images to classify their chemical content.")
        st.write(f"Maximum number of uploading images at once: {MAX_UPLOAD_IMAGES}")

        uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg", "tiff", "tif"], accept_multiple_files=True)

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        if selected_example_images:
            st.session_state.uploaded_files.extend([img for img in example_images if img.name in selected_example_images])

        if st.session_state.uploaded_files:
            if len(st.session_state.uploaded_files) > MAX_UPLOAD_IMAGES:
                st.error(f"Maximum number of uploading images reached. Only {MAX_UPLOAD_IMAGES} images will be processed.")
            with st.spinner('Processing images...'):
                results = classify_multiple_images(st.session_state.uploaded_files)
                if results:
                    st.session_state.results = results

    elif current_mode == "Input Image Path: Local Server Run":
        st.write("For local server use only: Provide an image path to classify the chemical content of image files.")
        image_path = st.text_input("Enter the image path:")
        if st.button("Classify Images"):
            st.session_state.results = []
            with st.spinner('Processing image path...'):
                result = classify_image_from_path(image_path)
                if result:
                    st.session_state.results = result

    if st.session_state.results:
        st.write("Classification Results:")

        classification_results = []
        for result in st.session_state.results:
            image_id = result.get('image_id')
            predicted_label = result.get('predicted_label', 'no chemical structures')
            classifier_package = result.get('classifier_package', 'ChemIC-ml_1.3')
            classifier_model = result.get('classifier_model', 'ResNet_50')

            image_data = Image.open(BytesIO(base64.b64decode(result.get('image_preview'))))

            # Encode image to Base64
            image_base64 = encode_image_to_base64(image_data)
            image_preview = f'<img src="data:image/png;base64,{image_base64}" width="100"/>'

            classification_results.append({
                'image_id': image_id,
                'image_preview': image_preview,
                'predicted_label': predicted_label,
                'classifier_package': classifier_package,
                'classifier_model': classifier_model,
                'image_base64': image_base64,
            })

        # Create DataFrame and show results
        feedback_df = pd.DataFrame(classification_results)

        display_df = feedback_df.drop('image_base64', axis=1)  # Drop base64 column for display

        # Display DataFrame with image previews
        html = display_df.to_html(escape=False, index=True)  # Ensure HTML is not escaped
        st.markdown(html, unsafe_allow_html=True)

        # Create CSV download link
        create_csv_download_link(df=display_df)

        # Ask if all predictions are correct
        all_correct = st.radio("Are the predictions correct?", ["Yes", "No"])

        if all_correct == "No":
            # Define the available options for correction
            labels = [
                'single chemical structure',
                'chemical reactions',
                'no chemical structures',
                'multiple chemical structures']

            # User feedback for each image
            corrected_labels = []
            for i, result in enumerate(st.session_state.results):
                image_id = result.get('image_id')
                predicted_label = result.get('predicted_label', 'no chemical structures')


                corrected_label = st.selectbox(
                    f"Correct label for {image_id}:",
                    options=[
                        predicted_label,  # Default selection is the predicted label
                        # Show all options except the predicted label
                        *[label for label in labels if label!=predicted_label]
                            ],
                    index=0,
                    key=f"feedback_{i}"
                )
                corrected_labels.append(corrected_label)

            # Update DataFrame with corrected y
            feedback_df['corrected_label'] = corrected_labels

            # Save feedback to DuckDB if corrected y are present
            if (feedback_df['predicted_label'] != feedback_df['corrected_label']).any():

                # FIXME: Implement this part to save to DuckDB, fix encoding to base64 string
                # save_to_duckdb(feedback_df)
                st.write("Thanks for your feedback! Your corrected y have been sent to the developers' team.")

    if st.sidebar.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/healthcheck")
            if response.status_code == 200:
                st.sidebar.success("Server is up and running!")
            else:
                st.sidebar.error("Server is not healthy.")
        except Exception as e:
            st.sidebar.error(f"Error connecting to API: {e}")

def main():

    st.set_page_config(
        page_title="ChemIC: Chemical Image Classifier",
        page_icon=":test_tube:",
        layout="wide",
    )

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(parent_dir, "public/ChemIC.svg")

    # Navigation and logo
    st.sidebar.image(logo_path, width=200)


    pages = [
        "Home",
        "About",
        "API Documentation",  # Add the new page
        "GitHub",
    ]

    urls = {"GitHub": "https://github.com/alexey-krasnov/ChemIC.git"}

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
    try:
        if not is_uvicorn_running():
            backend_process = start_uvicorn()
        else:
            print("Uvicorn is already running.")
    except Exception as e:
        print(f"Error starting backend server: {e}")
    else:
        main()