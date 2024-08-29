# import streamlit as st
# import requests
# import base64
# from PIL import Image
# from io import BytesIO
# import pandas as pd
# from datetime import datetime
#
# # Define your API URL
# API_URL = "http://127.0.0.1:5000"  # Update with your actual API endpoint if different
#
# def show_footer():
#     """
#     Display a fixed footer at the bottom of the Streamlit app.
#     """
#     st.markdown(
#         """
#         <style>
#         .footer {
#             position: fixed;
#             left: 0;
#             bottom: 0;
#             width: 100%;
#             background-color: #f2f2f2;
#             color: black;
#             text-align: center;
#             padding: 10px;
#             z-index: 999;
#         }
#         </style>
#         <div class="footer">
#             <p>&copy; 2024 Chemical Image Classifier. All rights reserved.</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#
# def classify_image_from_file(image_file):
#     """
#     Send an image file to the backend for classification.
#     """
#     try:
#         img = Image.open(image_file)
#         buffered = BytesIO()
#         img.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#
#         # Send the image data to the API
#         response = requests.post(f"{API_URL}/classify_image", data={"image_data": img_str})
#
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Error: {response.json().get('error')}")
#             return None
#     except Exception as e:
#         st.error(f"Error processing the image: {e}")
#         return None
#
# def classify_multiple_images(image_files):
#     """
#     Send multiple image files to the backend for classification.
#     """
#     results = []
#     for image_file in image_files:
#         result = classify_image_from_file(image_file)
#         if result:
#             results.append(result)
#     return results
#
# def classify_image_from_path(image_path):
#     """
#     Send an image path to the backend for classification.
#     """
#     try:
#         response = requests.post(f"{API_URL}/classify_image", data={"image_path": image_path})
#
#         if response.status_code == 200:
#             return response.json()
#         else:
#             st.error(f"Error: {response.json().get('error')}")
#             return None
#     except Exception as e:
#         st.error(f"Error processing the image path: {e}")
#         return None
#
# def create_csv_download_link(df):
#     """
#     Create a download link for the DataFrame as a CSV file.
#     """
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     csv = df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="Download CSV",
#         data=csv,
#         file_name=f"classification_results_{timestamp}.csv",
#         mime="text/csv",
#     )
#
# def main():
#     """
#     Main function for running the Streamlit app.
#     """
#     st.set_page_config(
#         page_title="Chemical Image Classifier",
#         page_icon=":test_tube:",
#         layout="wide",
#     )
#
#     st.title("Chemical Image Classifier")
#     st.write("Upload one or more images or provide an image path to classify their chemical content.")
#
#     st.sidebar.header("Options")
#     mode = st.sidebar.radio("Select Input Mode", ["Upload Images", "Input Image Path"])
#
#     if mode == "Upload Images":
#         uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
#         if uploaded_files:
#             results = classify_multiple_images(uploaded_files)
#             if results:
#                 st.write("Classification Results:")
#
#                 # Combine all results into a single DataFrame
#                 combined_df = pd.DataFrame(results)
#
#                 # Display DataFrame
#                 st.dataframe(results)
#
#                 # Provide download link for CSV
#                 create_csv_download_link(combined_df)
#
#     elif mode == "Input Image Path":
#         image_path = st.text_input("Enter the image path:")
#         if st.button("Classify Image"):
#             results = classify_image_from_path(image_path)
#             if results:
#                 st.write("Classification Results:")
#
#                 # Convert the result into a DataFrame
#                 result_df = pd.DataFrame(results)
#
#                 # Display DataFrame
#                 st.dataframe(results)
#
#                 # Provide download link for CSV
#                 create_csv_download_link(result_df)
#
#     # Health Check
#     if st.sidebar.button("Check API Health"):
#         try:
#             response = requests.get(f"{API_URL}/healthcheck")
#             if response.status_code == 200:
#                 st.sidebar.success("API is up and running!")
#             else:
#                 st.sidebar.error("API is not healthy.")
#         except Exception as e:
#             st.sidebar.error(f"Error connecting to API: {e}")
#
#     show_footer()
#
# if __name__ == "__main__":
#     main()
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import pandas as pd
from datetime import datetime

# Define your API URL
API_URL = "http://127.0.0.1:5000"  # Update with your actual API endpoint if different

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
    try:
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        response = requests.post(f"{API_URL}/classify_image", data={"image_data": img_str})

        # Inspect the raw response to see what it looks like
        st.write("API Response:", response.json())

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                result['image_id'] = image_file.name
                return result
            elif isinstance(result, list) and len(result) > 0:
                result[0]['image_id'] = image_file.name
                return result[0]  # Assuming we're dealing with a single result for this example
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

        # Inspect the raw response to see what it looks like
        st.write("API Response:", response.json())

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict):
                result['image_id'] = image_path
                return result
            elif isinstance(result, list) and len(result) > 0:
                result[0]['image_id'] = image_path
                return result[0]  # Assuming we're dealing with a single result for this example
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"classification_results_{timestamp}.csv",
        mime="text/csv",
    )

def main():
    st.set_page_config(
        page_title="Chemical Image Classifier",
        page_icon=":test_tube:",
        layout="wide",
    )

    st.title("Chemical Image Classifier")
    st.write("Upload one or more images or provide an image path to classify their chemical content.")

    st.sidebar.header("Options")
    mode = st.sidebar.radio("Select Input Mode", ["Upload Images", "Input Image Path"])

    if mode == "Upload Images":
        uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files:
            results = classify_multiple_images(uploaded_files)
            if results:
                st.write("Classification Results:")

                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'image_id': result.get('image_id'),
                        'predicted_label': result.get('predicted_label', 'no chemical structures'),
                        'program': result.get('program', 'ChemIC'),
                        'program_version': result.get('program_version', '1.2')
                    })

                combined_df = pd.DataFrame(formatted_results)
                st.dataframe(combined_df)
                create_csv_download_link(combined_df)

    elif mode == "Input Image Path":
        image_path = st.text_input("Enter the image path:")
        if st.button("Classify Image"):
            result = classify_image_from_path(image_path)
            if result:
                st.write("Classification Results:")

                formatted_result = {
                    'image_id': result.get('image_id'),
                    'predicted_label': result.get('predicted_label', 'no chemical structures'),
                    'program': result.get('program', 'ChemIC'),
                    'program_version': result.get('program_version', '1.2')
                }

                result_df = pd.DataFrame([formatted_result])
                st.dataframe(result_df)
                create_csv_download_link(result_df)

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
