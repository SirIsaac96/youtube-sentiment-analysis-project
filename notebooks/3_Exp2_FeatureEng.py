
# Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import mlflow
import mlflow.sklearn
import dagshub


# MLFlow setup
dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-sentiment-analysis-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow')
mlflow.set_experiment('Exp 2 - BoW vs TF-IDF Feature Engineering')


# Step 1: Load and preprocess the data
data = pd.read_csv('data/interim/youtube_sentiments.csv')


# Drop missing and duplicate values
data.dropna(inplace = True)
data.drop_duplicates(inplace = True)


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

    # 9. Tokenization
    tokens = text.split()

    # 10. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


data['clean_text'] = data['text'].apply(preprocess_text)


# Remove rows with empty cleaned_text
data = data[~(data['clean_text'].str.strip() == '')]


# Step 2: Feature Engineering
x = data['clean_text']
y = data['sentiment']


# Step 3: Function to run the experiment with different feature extraction methods
def run_experiment(vectorizer_type, ngram_range, vectorizer_max_features, vectorizer_name):
    # Feature extraction
    if vectorizer_type == 'BoW':
        vectorizer = CountVectorizer(max_features = vectorizer_max_features, ngram_range = ngram_range)
    elif vectorizer_type == 'TF-IDF':
        vectorizer = TfidfVectorizer(max_features = vectorizer_max_features, ngram_range = ngram_range)
    else:
        raise ValueError("Unsupported vectorizer type")
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42, stratify = y
    )

    # Model training with Random Forest
    with mlflow.start_run(run_name=f"Exp2_{vectorizer_name}_ngrams_{ngram_range}") as run:
        # Log a description for the run
        mlflow.set_tag('author', 'Isaac-Otom')
        mlflow.set_tag("model", "Exp2 - Feature Engineering")
        mlflow.set_tag('model_type', 'Random-Forest')
        mlflow.set_tag('description', f'Random Forest model for sentiment analysis using {vectorizer_name} features with ngram range {ngram_range}, and max features {vectorizer_max_features}')

        # Log parameters for the vectorizer
        mlflow.log_param('vectorizer_type', vectorizer_name)
        mlflow.log_param('ngram_range', ngram_range)
        mlflow.log_param('vectorizer_max_features', vectorizer_max_features)

        # Log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        # Log parameters
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        # Train the model
        model = RandomForestClassifier(
            n_estimators = n_estimators, 
            max_depth = max_depth, 
            random_state = 42
        )

        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("model", model)
        ])
        pipeline.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = pipeline.predict(x_test)

        # Log metrics
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', acc)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict = True)
        for cls, metrics in class_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{cls.replace(' ', '_')}_{metric_name}", metric_value)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix: {vectorizer_name}, {ngram_range}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Log the confusion matrix plot
        mlflow.log_figure(
            fig,
            artifact_file=f"confusion_matrices/{vectorizer_name}/cm_{ngram_range}.png"
        )
        plt.close(fig)

        # Log the trained model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="Model"
        )

        # Log this script (2_Exp1_BaselineModel.py) as an artifact
        try:
            mlflow.log_artifact(__file__)
        except Exception:
            pass


# Step 4: Run experiments for BoW and TF-IDF with different n-gram ranges
ngram_ranges = [(1, 1), (1, 2), (1, 3)] # Unigrams, Bigrams, Trigrams
max_features = 5000


for ngram_range in ngram_ranges:
    # Run experiment with Bag of Words
    run_experiment('BoW', ngram_range, max_features, 'BoW')

    # Run experiment with TF-IDF
    run_experiment('TF-IDF', ngram_range, max_features, 'TF-IDF')


print('Training complete. Logged to MLFlow successfully.')
