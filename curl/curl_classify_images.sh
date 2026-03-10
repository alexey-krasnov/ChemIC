#!/bin/bash
# Send a POST request containing base64 encoded data to a server using curl for image classification

# Usage:
# ./curl_classify_images.sh --image_path <image_path>
# ./curl_classify_images.sh --image_data <image_data>

#Examples:
# Send image_path:
# ./curl_classify_images.sh --image_path /path/to/image.png
# Send image_data (base64-encoded):

# ./curl_classify_images.sh --image_data /path/to/image.png
# Replace /path/to/image.png with the path to your image file. Adjust the server URL (http://127.0.0.1:5010/classify_images) as needed based on your server configuration.

# Author: Dr. Aleksei Krasnov
# a.krasnov@digital-science.com
# OntoChem GmbH part of the Digital Science


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --image_path)
        image_path="$2"
        shift # past argument
        shift # past value
        ;;
        --image_data)
        image_data="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

if [[ -n "$image_path" && -n "$image_data" ]]; then
    echo "Error: Both image_path and image_data cannot be specified simultaneously."
    exit 1
fi

if [[ -n "$image_path" ]]; then
    # Send a POST request with image_path
    curl -X POST -F "image_path=$image_path" http://127.0.0.1:5010/classify_images
    echo "Sent image_path: $image_path"
elif [[ -n "$image_data" ]]; then

    # Base64 encode the image data on Linux
    #base64_encoded_data=$(base64 -w 0 "$image_data")

    # Base64 encode the image data (OS-specific command)
    base64_encoded_data=$(base64 -i "$image_data")

    # Send a POST request with image_data
    curl -X POST -F "image_data=$base64_encoded_data" http://127.0.0.1:5010/classify_images
    echo "Sent image_data"
else
    echo "Error: Please specify either --image_path or --image_data."
    exit 1
fi
