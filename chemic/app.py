import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel

from chemic.image_classifier import ImageClassifier

# Start time to calculate loading time
start = time.time()

# Initialize FastAPI app
app = FastAPI()

# End time to calculate loading time
end = time.time()
print(f'ChemIC web service {__name__} is ready to work...')
print(f"Launching took {time.strftime('%H:%M:%S', time.gmtime(end - start))}")

# Creating an instance of Chemical ImageClassifier
image_classifier = ImageClassifier()

class ClassificationResult(BaseModel):
    image_id: Optional[str]
    predicted_label: Optional[str]
    classifier_package: Optional[str]
    classifier_model: Optional[str]

@app.post("/classify_image", response_model=List[ClassificationResult])
async def classify_image(image_path: Optional[str] = Form(None), image_data: Optional[str] = Form(None)):
    try:
        image_classifier.results = []  # Reset results for each classification cycle

        if image_path:
            print(f'Server received image {image_path}')
            image_classifier.send_to_classifier(image_path=image_path)
        elif image_data:
            print('Server received image data')
            image_classifier.process_image_data(base64_data=image_data)
        else:
            raise HTTPException(status_code=400, detail="Neither image_path nor image_data provided.")

        # Ensure results are returned in the correct format
        results = image_classifier.results
        print(results)
        print(f"Total number of images processed: {image_classifier.total_number_images}")
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during classification: {str(e)}")

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "Server is up and running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5010)
