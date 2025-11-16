# 🚀 NASA Space Apps - Exoplanet AI Analyst

A cutting-edge machine learning application that classifies Kepler Object of Interest (KOI) candidates as exoplanets using advanced AI techniques.

## 🌟 Features

- **Single Analysis**: Analyze individual exoplanet candidates with detailed form input
- **Batch Analysis**: Upload CSV/Excel files for bulk exoplanet classification
- **AI Chat**: Interactive chatbot powered by Gemini AI for detailed explanations
- **Beautiful UI**: Dark, technological theme with smooth animations
- **Real-time Predictions**: Instant classification with confidence scores

## 🔬 Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, LightGBM, Stacking Ensemble
- **AI Chat**: Google Gemini API
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Render

## 🚀 Live Demo

[https://render.com](https://nasa-spaceapps-exoplanet-1.onrender.com/))

## 📊 Model Performance

- **Algorithm**: Stacking Ensemble (LightGBM + Gradient Boosting + Random Forest)
- **Accuracy**: High-performance classification model
- **Features**: 43 astronomical parameters including orbital period, transit depth, stellar properties

## 🛠️ Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (create `.env` file)
4. Run: `python app.py`
5. Visit: `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                 # Flask application
├── index.html            # Main web interface
├── script.js             # Frontend JavaScript
├── style.css             # Styling and animations
├── train_and_save_model.py # ML model training
├── requirements.txt      # Python dependencies
├── Procfile             # Deployment configuration
└── *.pkl                # Trained ML models
```

## 🌌 NASA Space Apps Challenge

This project was developed for the NASA Space Apps Challenge, focusing on exoplanet discovery and classification using machine learning and AI technologies.

## 📄 License

MIT License - Feel free to use for educational and research purposes.
