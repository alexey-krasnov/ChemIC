import base64
import hashlib
import importlib.metadata


def generate_unique_identifier(image_path=None, base64_encoded_image=None, np_array=None):
    """
    Generate a unique identifier for an image.

    Parameters:
    - image_path: Path to the image file.
    - base64_encoded_image: Base64 encoded string of the image.
    - np_array: NumPy array representing the image.

    Returns:
    - unique_identifier: A SHA-256 hash string of the image.
    """
    # Ensure one and only one argument is provided
    if sum(x is not None for x in [image_path, base64_encoded_image, np_array]) != 1:
        raise ValueError("You must provide either image_path, base64_encoded_image, "
                         "or np_array, but not more than one.")

    # If image_path is provided, read and encode the image
    if image_path:
        with open(image_path, "rb") as image_file:
            base64_encoded_image = base64.b64encode(image_file.read())

    # If np_array is provided, convert to bytes and encode as base64
    elif np_array is not None:
        np_array_bytes = np_array.tobytes()
        base64_encoded_image = base64.b64encode(np_array_bytes)

    # Hash the base64 encoded image data
    hash_object = hashlib.sha256(base64_encoded_image)
    unique_identifier = hash_object.hexdigest()
    return unique_identifier


def get_package_version(package_name: str) -> str:
    """
    Get the version of the specified Python package.

    Parameters:
        package_name (str): The name of the Python package.

    Returns:
        str: The version of the specified package if installed.
             If the package is not installed, returns a message indicating that the package is not installed.
    """
    try:
        package_version = importlib.metadata.version(package_name)
        return package_version
    except importlib.metadata.PackageNotFoundError:
        return f"{package_name} is not installed"


def get_package_name_version(package_name: str) -> str:
    """
    Get the combined package name and version string.

    Parameters:
        package_name (str): The name of the Python package.

    Returns:
        str: The combined package name and version string.
             If the package is not installed, returns a message indicating that the package is not installed.
    """
    version = get_package_version(package_name)
    if version == f"{package_name} is not installed":
        return version
    return f"{package_name}_{version}"
