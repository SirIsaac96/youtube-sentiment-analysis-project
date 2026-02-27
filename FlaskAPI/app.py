
# libraries
import io
import os
import re
import logging
import pickle
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt-sentiment-analysis-api")


# Ensure required NLTK data is available
def ensure_nltk_data():
    try:
        stopwords.words('english')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        logger.info("Downloading missing NLTK data (stopwords, wordnet, omw-1.4)...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    except Exception as e:
        logger.exception("Error ensuring NLTK data: %s", e)
        raise


ensure_nltk_data()


# Flask app 
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# Preprocessing
# Global NLP objects (load once)
STOP_WORDS = set(stopwords.words('english')) - {
    'not', 'no', 'but', 'however', 'yet', 'you'
}
LEMMATIZER = WordNetLemmatizer()
IMPORTANT_SHORT = {"not", "bad", "yes", "wow", "fun", "win"}


# Preprocess a single comment
def preprocess_comment(text):
    try:
        if not text:
            return ""

        # 1. Convert to lowercase
        text = str(text).lower()

        # 2. Remove mentions (@user123)
        text = re.sub(r'@\w+', '', text)

        # 3. Remove newline characters
        text = re.sub(r'\n+', ' ', text)

        # 4. Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # 6. Emoji handling
        text = emoji.demojize(text, delimiters=(' ', ' '))

        # 5. Remove punctuations, numbers, and special chars except hashtags
        text = re.sub(r'[^a-zA-Z#\s]', '', text)

        # 6. Remove stopwords and short words (less than 3 characters) while keeping important short words
        words = [
            w for w in text.split()
            if (w not in STOP_WORDS) and (len(w) >= 3 or w in IMPORTANT_SHORT)
        ]

        lemmatized = [LEMMATIZER.lemmatize(w) for w in words]

        return " ".join(lemmatized)

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        return ""


'''
# Model loading from MLFlow registry
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    logger.info("Loading model from MLflow registry...")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    # Initialize DagsHub and MLflow client
    dagshub.init(
        repo_owner = "SirIsaac96",
        repo_name = "youtube-sentiment-analysis-project",
        mlflow = True
    )
    mlflow.set_tracking_uri(
        "https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow"
    )

    client = MlflowClient()

    # Construct model URI using numeric version directly
    model_uri = f"models:/{model_name}/{model_version}"
    logger.info("Loading model from URI: %s", model_uri)

    # Load the model from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully from MLflow registry.")

    # Load the vectorizer from the local file system
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Model and vectorizer loaded successfully.")

    return model, vectorizer


# Initialize model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer(
        'YouTube_Sentiment_Analysis_Model',
        '2',
        './bow_vectorizer.pkl'
    )
except Exception as e:
    logger.exception("Failed to load model/vectorizer: %s", e)
    model, vectorizer = None, None
'''


# Function to load model and vectorizer
def load_model_and_vectorizer(model_path, vectorizer_path):
    logger.info("Loading model and vectorizer...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    # Load the model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully.")

    # Load the vectorizer
    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)
    logger.info("Vectorizer loaded successfully.")

    return model, vectorizer


# Initialize model and vectorizer
try:
    model, vectorizer = load_model_and_vectorizer(
        "./models/lgbm_model.pkl",
        "./models/bow_vectorizer.pkl"
    )
except Exception as e:
    logger.exception("Failed to load model/vectorizer: %s", e)
    model, vectorizer = None, None


# Routes
@app.route("/")
def home():
    return "YouTube Sentiment Analysis API (Flask) - Running"


# Predict Sentiment
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model/vectorizer not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' field."}), 400

    comments = data["comments"]
    if not isinstance(comments, list) or not comments:
        return jsonify({"error": "'comments' must be a non-empty list."}), 400

    try:
        # 1. Preprocess
        preprocessed = [preprocess_comment(c) for c in comments]

        # 2. Vectorize
        transformed = vectorizer.transform(preprocessed).astype("float32")

        # 3. Convert to DataFrame for model input
        transformed_df = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # 4. Predict
        preds = model.predict(transformed_df)

        preds = [int(p) if hasattr(p, "item") else int(p) for p in preds]

        return jsonify(
            [{"comment": c, "sentiment": p} for c, p in zip(comments, preds)]
        )

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Predict with timestamps
@app.route("/predict_with_timestamps", methods=["POST"])
def predict_with_timestamps():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model/vectorizer not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "comments" not in data:
        return jsonify({"error": "Missing 'comments' field."}), 400

    comments_data = data["comments"]
    if not isinstance(comments_data, list):
        return jsonify({"error": "'comments' must be a list of objects."}), 400

    try:
        texts = [item.get("text", "") for item in comments_data]
        timestamps = [item.get("timestamp") for item in comments_data]

        # 1. Preprocess
        preprocessed = [preprocess_comment(t) for t in texts]

        # 2. Vectorize
        transformed = vectorizer.transform(preprocessed).astype("float32")

        # 3. Convert to DataFrame for model input
        transformed_df = pd.DataFrame(
            transformed.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # 4. Predict
        preds = model.predict(transformed_df)
        preds = [int(p) if hasattr(p, "item") else int(p) for p in preds]
        response = [
            {"comment": t, "sentiment": p, "timestamp": ts}
            for t, p, ts in zip(texts, preds, timestamps)
        ]
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction with timestamps error: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# Generate Pie Chart
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    try:
        data = request.get_json(force=True)
        sentiment_counts = data.get("sentiment_counts", {})
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            return jsonify({"error": "Sentiment counts sum to zero"}), 400

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "white"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating chart: %s", e)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# Generate Wordcloud
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    try:
        data = request.get_json(force=True)
        comments = data.get("comments", [])
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=STOP_WORDS,
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating wordcloud: %s", e)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# Generate Sentiment Trend Graph
@app.route("/generate_trend_graph", methods=["POST"])
def generate_trend_graph():
    try:
        data = request.get_json(force=True)
        sentiment_data = data.get("sentiment_data", [])
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        if "timestamp" not in df or "sentiment" not in df:
            return jsonify({"error": "Missing required keys (timestamp, sentiment)."}), 400

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        plt.figure(figsize=(12, 6))
        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker="o",
                linestyle="-",
                label=labels[val],
                color=colors[val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        logger.exception("Error generating trend graph: %s", e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


# Run the app
if __name__ == "__main__":
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
