# ===================
# Halal vs Haram Food Classifier

A deep learning model to classify food images as **Halal** or **Haram** using EfficientNet.

## Features
- EfficientNetB0-based image classifier
- Dataset with images and `labels.csv` for halal/haram labeling
- Prediction script for new images

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python model.py
```

## Prediction
```bash
python predict.py path/to/food.jpg
```

## Dataset Format
Images are organized in folders `halal/` and `haram/`, and accompanied by a `labels.csv` file containing:
- `filename`
- `label`
- `description`
- `ingredients`

## Future Work
- NLP classification using food ingredients or descriptions
- Multilingual support for labeling
