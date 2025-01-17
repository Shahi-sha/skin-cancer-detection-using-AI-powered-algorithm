from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os 
import uvicorn 
import cv2 
import numpy as np 
from PIL import Image 
import io 
from fastapi import Request 
import base64 
import json 


import keras

model  = keras.models.load_model('my_skin_disease_pred_model.h5')


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def load_data():
    with open(os.path.join("static", "data.json"), "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def predict_image(image_data: bytes):
    image = Image.open(io.BytesIO(image_data))

    
    if image.mode != 'RGB':
        image = image.convert('RGB')

    
    image = np.array(image)
    
    image = cv2.resize(image, (64, 64))  
    image = image.astype('float32') / 255.0  

    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_prob = predictions[0][predicted_class_index]
    predicted_class_name = data['class_names']['en'][f"{predicted_class_index}"]

    return predicted_class_name, predicted_prob, predicted_class_index


@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    
    image_data = await file.read()

    
    predicted_class_name, predicted_prob, predicted_class_index = predict_image(image_data)
    
    image = Image.open(io.BytesIO(image_data))
    target_height = 400
    image = image.resize((int(target_height * (image.width / image.height)), target_height))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "picture": img_str, 
            "predicted_class_index": predicted_class_index, 
            "predicted_prob": f"{predicted_prob:.2%}",
        }
    )

if __name__ == '__main__':
    
    data = load_data()
    uvicorn.run(app, host='0.0.0.0', port=8000) 
