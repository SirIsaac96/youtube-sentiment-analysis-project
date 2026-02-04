
# Libraries
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import mlflow
import dagshub
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# MLflow / DagsHub setup
dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-sentiment-analysis-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow')
mlflow.set_experiment("Exp 7 - Stacking (LightGBM + Logistic Regression + KNN)")


# Step 1: Load and preprocess the data
data = pd.read_csv('data/interim/youtube_sentiments.csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)


# Text preprocessing function
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove mentions (@user123)
    text = re.sub(r'@\w+', '', text)

    # 3. Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # 4. Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 5. Remove punctuations, numbers, and special chars except hashtags
    text = re.sub(r'[^a-zA-Z#\s]', '', text)

    # 6. Emoji handling
    text = emoji.demojize(text, delimiters=(' ', ' '))

    # 7. Remove stopwords
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'but', 'however', 'yet', 'you'}
    text = ' '.join([w for w in text.split() if w not in stop_words])

    # 8. Remove short words (less than 3 characters)
    important_short = {"not", "bad", "yes", "wow", "fun", "win"}
    text = ' '.join([w for w in text.split() if len(w) >= 3 or w in important_short])

    # 9. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()]

    return ' '.join(tokens)


data["clean_text"] = data["text"].apply(preprocess_text)


# Remap the class labels from [-1, 0, 1] to [2, 0, 1]
data["sentiment"] = data["sentiment"].map({-1: 2, 0: 0, 1: 1})


# Remove any rows with empty cleaned text
data = data[data["clean_text"].str.strip() != ""]


# Step 2: Feature Engineering and vectorization
x_cleaned = data["clean_text"]
y_cleaned = data["sentiment"]


# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x_cleaned, y_cleaned, test_size=0.2, random_state=42, stratify=y_cleaned
)


# Vectorization
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
x_train_bow = vectorizer.fit_transform(x_train).astype('float32')
x_test_bow = vectorizer.transform(x_test).astype('float32')


# Base learners
lightgbm_model = LGBMClassifier(
    objective = 'multiclass',
    num_class = 3,
    metric = 'multi_logloss',
    class_weight = 'balanced',
    reg_alpha = 0.1,
    reg_lambda = 0.1,
    learning_rate = 0.28693034784335925,
    n_estimators = 100,
    max_depth = 21,
    random_state = 42
)


# Logistic Regression
log_reg_model = LogisticRegression(
    max_iter = 1000,
    class_weight = 'balanced',
    solver = 'lbfgs',
    multi_class = 'multinomial',
    random_state = 42
)


# K-Nearest Neighbors as meta learner
knn_learner_model = KNeighborsClassifier(n_neighbors = 5)


# Stacking Classifier, with LightGBM and Logistic Regression as base learners and KNN as meta learner
stacking_classifier = StackingClassifier(
    estimators = [
        ('lightgbm', lightgbm_model),
        ('logistic_regression', log_reg_model)
    ],
    final_estimator = knn_learner_model,
    cv = 5,
    n_jobs = -1,
    passthrough = True
)


# Step 3: Log results to MLFlow
with mlflow.start_run(run_name="Stacking_LGBM_LR_KNN_BoW"):

    # Tags
    mlflow.set_tag("model_type", "Stacking Classifier")
    mlflow.set_tag("features", "BoW (1,3 grams)")
    mlflow.set_tag("task", "Multiclass Sentiment Classification")

    # Log vectorizer params
    mlflow.log_param("ngram_range", "(1,3)")
    mlflow.log_param("max_features", 1000)

    # Log LightGBM params
    mlflow.log_params({
        "lgb_learning_rate": 0.24412363272845405,
        "lgb_n_estimators": 100,
        "lgb_max_depth": 30,
        "lgb_reg_alpha": 0.1,
        "lgb_reg_lambda": 0.1
    })

    # Log Logistic Regression params
    mlflow.log_params({
        "lr_solver": "lbfgs",
        "lr_max_iter": 1000
    })

    # Log KNN params
    mlflow.log_param("knn_neighbors", 5)

    # Train
    stacking_classifier.fit(x_train_bow, y_train)

    # Predict
    y_pred = stacking_classifier.predict(x_test_bow)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1_weighted)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            for m_name, m_val in metrics.items():
                mlflow.log_metric(f"{cls}_{m_name}", m_val)

    # Log model
    mlflow.sklearn.log_model(
        stacking_classifier,
        artifact_path = "stacking_model"
    )

    print(classification_report(y_test, y_pred))
