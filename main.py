from fastapi import FastAPI, File, UploadFile
from model import HalalHaramClassifier
from utils import transform_image
import torch

app = FastAPI()

# Load model
model = HalalHaramClassifier()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    tensor = transform_image(image_bytes)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        label = "halal" if predicted.item() == 0 else "haram"

    return {"prediction": label}
