## 🕌 𝗜𝗻𝘀𝗽𝗶𝗿𝗲𝗱 𝗯𝘆 𝘁𝗵𝗲 𝗤𝘂𝗿𝗮𝗻:
```bash
وَكُلُوا مِمَّا رَزَقَكُمُ اللَّهُ حَلَالًا طَيِّبًا ۚ وَاتَّقُوا اللَّهَ الَّذِي أَنتُم بِهِ مُؤْمِنُونَ
"And eat of what Allah has provided for you [which is] lawful and good. And fear Allah, in whom you are believers."
```
— 𝗤𝘂𝗿𝗮𝗻 𝟱:𝟴𝟴


# Halal vs Haram Food Classifier

A deep learning model to classify food images as **Halal** or **Haram** using EfficientNet.

## Features
- EfficientNetB0-based image classifier
- Dataset with images and `labels.csv` for halal/haram labeling
- Prediction script for new images
  
## 📦 Datasets (Help Improve Accuracy!)
- Image Dataset: Kaggle - Halal/Haram Images https://www.kaggle.com/datasets/hamzamehar/data-set-halal-and-haram

- CSV Labels: Kaggle - Halal/Haram Metadata  https://www.kaggle.com/datasets/hamzamehar/halal-vs-haram-csv-data

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

  ## 👥 Contributors  
<a href="https://github.com/Hamza-Rafique/halal-verification/graphs/contributors">  
  <img src="https://contrib.rocks/image?repo=Hamza-Rafique/halal-verification" />  
</a>  
