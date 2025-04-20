# Simple hardcoded example
HARAM_DATABASE = ["gelatin", "alcohol", "enzymes", "lard", "rum"]

def check_against_database(ingredients_text):
    found = []
    for haram in HARAM_DATABASE:
        if haram.lower() in ingredients_text.lower():
            found.append(haram)
    return found

def format_ingredients(raw_text):
    # You can split by lines or commas
    return [{"name": ing.strip(), "status": "questionable"} for ing in raw_text.split(',') if ing.strip()]

def calculate_halal_score(logo_detected, haram_ingredients):
    base = 100 if logo_detected else 70
    penalty = len(haram_ingredients) * 20
    return max(0, base - penalty)
