from fastapi import FastAPI, File, UploadFile
from sklearn.feature_extraction import image
import uvicorn
import numpy as np
import cv2

from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("C:\\Users\\HP\\Documents\\Winter_sem_2021_2022\\code1\\Corn-disease-classification\\saved_models\\new")
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