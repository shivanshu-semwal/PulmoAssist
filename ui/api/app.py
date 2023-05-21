from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import urllib.request

description = """
This API helps you classify Chest X-Rays images as **'Covid', 'Normal', 'TB', 'Viral_Pneumonia'**.

## How to use it?

- Send a **POST** request on our API with a JSON file containing url of the image you want to classify.
- Alternatively, try uploading the image directly on this page using try-it-out functionality in `predict_image` route.

## Response Format

It will return you 2 two things - 
1. `class` - [Covid, Normal, TB, Pneumonia] (String Datatype)
2. `class_probability` - The probability by which model is predicting that the image belongs to the given class (Float Datatype)
"""

app = FastAPI(
    title="PulmoAssist API",
    version="2.1",
    description=description,
    contact={
        "name": "PulmoAssist",
        "url": "https://github.com/shivanshu-semwal"
    }
)

interpreter = tf.lite.Interpreter(model_path='model/quantized_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']

labels = {
    0: 'Covid',
    1: 'Normal',
    2: 'TB',
    3: 'Pneumonia'
}

def preprocess_img(img):
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img, dtype='float32')
    # img = img / 255
    img = img.reshape((1, 256, 256, 3))
    return img

def classify_image(img):
    img = preprocess_img(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(predictions[0])
    result = {
        'class': labels[pred],
        'class_probability': np.round(predictions[0][pred] * 100, 2)
    }
    return result


class img_url(BaseModel):
    url: str

    class Config:
        schema_extra = {
            "example": {
                "url": "https://i.ibb.co/FBSztPS/0120.jpg"
            }
        }


@app.get("/ping")
async def ping():
    return "Working!"


@app.post(
    "/predict_image",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "class": "Covid",
                        "class_probability": 97.42
                    }
                }
            }
        }
    }
)
async def predict(
    file: UploadFile = File(...)
):
    image = Image.open(BytesIO(await file.read()))
    response = classify_image(image)
    return response


@app.post(
    "/",
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "class": "Pneumonia",
                        "class_probability": 99.94
                    }
                }
            }
        }
    }
)
async def classify_url(item: img_url):
    req = urllib.request.urlretrieve(item.url, "saved")
    image = Image.open("saved")
    response = classify_image(image)
    return response
