import io
import numpy as np
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import StreamingResponse
import torch
from torchvision import models,transforms
from PIL import Image,ImageDraw


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])


app = FastAPI()

def predict_and_draw(image : Image.Image):

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    img_rgb = image.convert("RGB")

    draw = ImageDraw.Draw(img_rgb)

    for box,score in zip(boxes,scores):
        if score>0.7:
            x_min,y_min,x_max,y_max = box
            draw.rectangle([x_min,y_min,x_max,y_max] , outline="red" , width=2)
    
    return img_rgb


@app.get("/")
def read_root():
    return {"message" : "Welcome to the Guns Object Detection API"}


@app.post("/predict/")
async def predict(file:UploadFile=File(...)):

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    output_image = predict_and_draw(image)

    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr , format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr , media_type="image/png")


