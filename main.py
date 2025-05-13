from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("model/halal_haram_classifier.h5")
IMAGE_SIZE = (224, 224)

class_labels = ["halal", "haram"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_labels[int(prediction[0][0] > 0.5)]
        confidence = float(prediction[0][0])

        return JSONResponse(content={
            "predicted": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
