# 🇧🇩 Bangla Fake News Detector

A Machine Learning based Bangla Fake News Detection system using BERT (Bidirectional Encoder Representations from Transformers).

##  Project Overview
This project aims to detect fake news in Bangla language using state-of-the-art Natural Language Processing techniques. The model is trained on a large dataset of Bangla news articles and can classify news as real or fake with high accuracy.

##  Features
-  **Single News Analysis** - Test individual Bangla news articles in real-time
- **Batch Processing** - Upload CSV files for bulk analysis of multiple news
- **Probability Visualization** - Interactive pie charts showing confidence scores
- **History Tracking** - Automatic record of all analyses with timestamps
- **Web Interface** - User-friendly Gradio web application

##  Project Structure

bangla-fake-news-detector/
│
├── data/ # Dataset directory (ignored by git)
├── models/ # Trained models (ignored by git)
├── src/ # Source code
│ ├── preprocess.py # Data preprocessing
│ ├── train.py # Model training
│ ├── predict.py # Prediction functions
│ └── init.py
├── web/ # Web application
│ ├── app.py # Gradio web app
│ ├── index.html # HTML template
│ └── style.css # CSS styling
├── requirements.txt # Python dependencies
├── README.md # Documentation
└── .gitignore # Git ignore file