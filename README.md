# YouTube Comments Sentiment Analysis System

[![CI/CD Pipeline](https://github.com/SirIsaac96/youtube-sentiment-analysis-project/actions/workflows/cicd.yaml/badge.svg)](https://github.com/SirIsaac96/youtube-sentiment-analysis-project/actions)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-blue)](https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blueviolet)](https://www.docker.com/)
[![Deployment](https://img.shields.io/badge/Render-Deployed-brightgreen)](https://youtube-sentiment-analysis-project.onrender.com/)

An end-to-end production-grade MLOps system that analyzes sentiment from educational YouTube comments in real-time.

The project implements the full ML lifecycle:
data collection → experimentation → reproducible pipeline → model registry → API serving → CI/CD → cloud deployment.

A Chrome Extension integrates directly with a deployed Flask API to deliver instant sentiment summaries on the YouTube watch page.

## Live Deployment

The Flask API is deployed on Render and serves real-time predictions.

You can test it using Postman or curl by sending a POST request with a JSON list of comments.

---

## Project Highlights

- Automated scraping of YouTube educational content comments
- NLP preprocessing & exploratory data analysis
- Multi-model experimentation with MLflow (via DagsHub)
- Reproducible ML pipeline using DVC
- Model registration & versioning
- REST API built with Flask
- Chrome Extension integration
- Dockerized deployment
- CI/CD using GitHub Actions
- Production deployment on Render

---

# System Architecture

## i. Training Pipeline Architecture

```
YouTube API Scraping
        ↓
Raw Dataset
        ↓
DVC Pipeline
 ├── Data Ingestion
 ├── Data Preprocessing
 ├── Model Training
 ├── Model Evaluation
 └── Model Registration
        ↓
Tracked Experiments (MLflow - DagsHub)
```

## ii. Production Inference Architecture

```
User Opens YouTube Video
        ↓
Chrome Extension Detects Video
        ↓
Extracts All Comments
        ↓
Sends Comments to Flask API
        ↓
Loaded Registered ML Model
        ↓
Sentiment Predictions
        ↓
Aggregated Sentiment Summary Displayed
```

---

# Project Structure

```
youtube-sentiment-analysis-project/
│
├── data/                        # DVC tracked datasets
├── models/                      # Serialized models
├── notebooks/                   # EDA & experimentation
├── src/
│   ├── data/                    # Data ingestion scripts
│   ├── features/                # Preprocessing logic
│   ├── model/                   # Training & evaluation logic
│   └── pipeline/                # DVC pipeline stages
│
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # Hyperparameters
├── FlaskAPI
│   └── app.py                   # Flask API entry point
├── requirements.txt
├── Dockerfile
├── .github/workflows/           # CI/CD workflows
└── README.md
```

---

# 1. Data Collection

- Scraped multiple educational YouTube channels  
- Iterated through channels → videos → comments  
- Used YouTube Data API  
- API keys and scraping logic excluded due to API security  

The goal was to collect educational-quality content comments for sentiment modeling.

---

# 2. Data Preprocessing & EDA

Performed in:

`1_Preprocessing_and_EDA.ipynb`

### Processing Steps

- Lowercasing
- Regex cleaning
- Emoji normalization
- Stopword removal
- Lemmatization
- Feature vectorization (BoW / TF-IDF)

### EDA Insights

- Class imbalance identification  
- Comment length distribution  
- Word frequency trends  
- Sentiment distribution analysis  

---

# 3. Model Experimentation & Tracking

Model development followed a structured, incremental experimentation strategy located in the `notebooks/` directory.

## Experiment Workflow

### Experiment 1 — Baseline Model  
`2_Exp1_BaselineModel.py`

- Random Forest Classifier
- Initial CountVectorizer setup
- Established baseline metrics

---

### Experiment 2 — Feature Engineering  
`3_Exp2_FeatureEng.py`

- Improved text normalization
- N-gram experimentation
- Vectorizer tuning
- Feature scaling refinement

Goal: Improve representation quality.

---

### Experiment 3 — Bag-of-Words Optimization  
`4_Exp3_BoW_13_MaxFeatures.py`

- Tuned `max_features`
- Vocabulary size optimization
- Dimensionality-performance trade-off analysis

---

### Experiment 4 — Handling Imbalanced Data  
`5_Exp4_Handling_ImbalData.py`

- Class weights
- Imbalance strategy comparisons
- Macro F1 prioritization

Key insight: Accuracy alone was misleading due to multiclass imbalance.

---

### Experiment 5 — XGBoost  
`6_Exp5_XGBoost.py`

- Gradient boosting classifier
- Hyperparameter tuning
- Compared to linear baseline

---

### Experiment 6 — LightGBM  
`7_Exp6_LightGBM.py`

- Faster gradient boosting
- Improved sparse feature handling
- Better multiclass performance

---

### Final Model — Stacking Ensemble  
`8_Stacking_Model.py`

Combined:

- Logistic Regression  
- KNN  
- LightGBM  

Meta-learner optimized for balanced multiclass performance.

This model achieved the best macro F1-score and generalization capability and was selected for deployment.

---

## Experiment Tracking

All experiments were tracked using MLflow hosted on DagsHub.

Tracked artifacts included:

- Model hyperparameters  
- Vectorizer configurations  
- Imbalance handling strategies  
- Evaluation metrics (Accuracy, Precision, Recall, Macro F1-score)  
- Serialized model artifacts  

This enabled:

- Structured comparison across experiments  
- Transparent model selection  
- Versioned experiment history  
- Reproducibility across environments    

---

# 4. Reproducible ML Pipeline

The workflow is orchestrated using:

- DVC (Data Version Control) 

Pipeline stages defined in `dvc.yaml`:

1. Data Ingestion  
2. Data Preprocessing  
3. Model Building  
4. Model Evaluation  
5. Model Registration  

To reproduce the pipeline:

```bash
dvc repro
```

---

# 5. Model Serving API

The trained model is served using:

- Flask (Python web framework)  

The API:

- Loads the registered model
- Accepts comment lists via POST request
- Returns predicted sentiments
- Aggregates overall sentiment summary

Run locally:

```bash
python FlaskAPI/app.py
```

---

# 6. Chrome Extension Integration

The Chrome extension:

- Detects active YouTube video
- Extracts comments dynamically
- Sends comments to deployed API
- Displays summarized sentiment in browser

Extension code excluded due to API security.

---

# 7. Containerization

Containerized using:

- Docker  

Build locally:

```bash
docker build -t yt-sentiment .
docker run -p 5000:5000 yt-sentiment
```

---

# 8. CI/CD Automation

Automated with:

- GitHub Actions  

Workflow includes:

- Automated testing
- Docker image build
- Deployment trigger

---

# 9. Production Deployment

API deployed on:

- Render (Cloud Platform)  

Render handles:

- Container deployment  
- Public API endpoint exposure  
- Automatic redeployments on push  

---

# Inference Design Considerations

The deployed model was selected not only for performance but also for:

- Balanced multiclass F1-score  
- Inference latency suitability for real-time browser integration  
- Model size constraints for container deployment  
- Stability across validation splits  

This ensured reliable real-time predictions when integrated with the Chrome Extension.

---

# Model Performance

Final model evaluated using:

- Accuracy  
- Precision  
- Recall  
- Macro F1-score (primary selection metric)

---

# Running the Project Locally

### Clone Repository

```bash
git clone https://github.com/SirIsaac96/youtube-sentiment-analysis-project
cd youtube-sentiment-analysis-project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Reproduce Pipeline

```bash
dvc repro
```

### Start API

```bash
python FlaskAPI/app.py
```

---

# Future Improvements

- Transformer-based models (BERT)
- Real-time streaming sentiment
- Cloud-native scaling
- Topic modeling integration

---

# Key Learnings

- Managing multiclass imbalance in NLP  
- Structured experiment tracking discipline  
- Reproducible ML workflows  
- API-based model serving  
- Secure API key management  
- Full-stack ML system integration  

---

# 👤 Author

Isaac O. Otom  
Machine Learning Engineer  
GitHub: https://github.com/SirIsaac96  

---

⭐ If you found this project interesting, consider giving it a star.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
