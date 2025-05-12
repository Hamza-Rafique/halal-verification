from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

model = load_model("halal_haram_classifier.h5")
IMG_SIZE = (224, 224)

img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
print("Prediction:", "Haram" if prediction > 0.5 else "Halal")