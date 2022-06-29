from fastapi import FastAPI, File, UploadFile
from sklearn.feature_extraction import image
import uvicorn
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#endpoint = "http://localhost:8605/v1/models/corn_model:predict"

MODEL = tf.keras.models.load_model("C:\\Users\\HP\\Documents\\Winter_sem_2021_2022\\code1\\Corn-disease-classification\\saved_models\\3")
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
@app.get("/ping")
async def ping():
    return "hello,i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image,0)
    #json_data = {
   #     "instances":image_batch.tolist()
   # }

    #response = requests.post(endpoint,json=json_data)
    #prediction = response.json()["predictions"][0]

    #predicted_class = class_names[np.argmax(prediction)]
    #confidence = np.max(prediction)
    
    #return{
     #   "class":predicted_class,
     #   "confidence":float(confidence)
   #}

    prediction = MODEL.predict(image_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    #dynamically route your traffic to different model
    return {
        'class':predicted_class,
        'confidence': float(confidence)
    }
     #if it takes two seconds to read the file instead of waiting
     #puts this function in suspend mode and second request can be served
     #need to convert these bytes to array

    

if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port=8000)