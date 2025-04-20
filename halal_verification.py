
from fastapi import FastAPI, UploadFile
from PIL import Image
import pytesseract
from utils.ai_model import load_ai_model, detect_halal_logo
from utils.ingredient_utils import check_against_database, format_ingredients, calculate_halal_score

app = FastAPI()

HALAL_LOGO_MODEL = load_ai_model()

@app.post("/verify")
async def verify_halal(file: UploadFile):
    image = Image.open(file.file)

    logo_detected = detect_halal_logo(image)
    ingredients_text = pytesseract.image_to_string(image)
    haram_ingredients = check_against_database(ingredients_text)

    return {
        "isHalal": len(haram_ingredients) == 0,
        "score": calculate_halal_score(logo_detected, haram_ingredients),
        "explanation": "Logo detected." if logo_detected else "No logo found. Score adjusted.",
        "ingredients": format_ingredients(ingredients_text),
        "transparencyScore": 85
    }
