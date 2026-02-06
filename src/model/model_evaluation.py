
# Libraries
import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature


# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Model evaluation functions
def read_data(file_path: str) -> pd.DataFrame:
    """Reads the CSV data file into a DataFrame."""
    try:
        # Read the data from the specified file path
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True) # Fill any NaN values with empty strings
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


def load_model(model_path: str):
    """Load the saved model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise


def load_vectorizer(vectorizer_path: str) -> CountVectorizer:
    """Load the saved CountVectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug(f"CountVectorizer loaded from {vectorizer_path}")
        return vectorizer

    except Exception as e:
        logger.error(f"Unexpected error while loading vectorizer: {e}")
        raise


def read_params(config_path: str) -> dict:
    """Reads the YAML configuration file."""
    try:
        # Read the YAML configuration file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        raise


def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return evaluation metrics."""
    try:
        # Make predictions
        y_pred = model.predict(x_test)

        # Calculate evaluation metrics
        acc_score = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.debug(f"Model evaluation completed.")

        return class_report, conf_matrix, acc_score

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def log_conf_matrix(conf_matrix, dataset_name):
    """Log confusion matrix as an artifact."""
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {dataset_name} Dataset')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Log the trained model
        mlflow.log_figure(fig, f'conf_matrix_{dataset_name}.png')
        plt.close(fig)
        logger.debug(f"Confusion matrix for {dataset_name} dataset logged successfully as an artifact.")

    except Exception as e:
        logger.error(f"Error logging confusion matrix: {e}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model information and evaluation metrics to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        # Save the dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(model_info, json_file, indent = 4)
        logger.debug(f"Model information saved to {file_path} successfully.")

    except Exception as e:
        logger.error(f"Error saving model information: {e}")
        raise


def main():
    # MLflow set up
    dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-sentiment-analysis-project', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow')
    mlflow.set_experiment('DVC Pipeline - Model Evaluation')

    with mlflow.start_run(run_name='Model Evaluation') as run:
        try:
            mlflow.set_tag("author", "Isaac-Otom")
            mlflow.set_tag("model_type", "LightGBM Classifier")
            mlflow.set_tag("dataset", "YouTube Comments Dataset")
            mlflow.set_tag("description", "Evaluation of the LightGBM model for sentiment analysis on YouTube comments")

            # Load parameters from the configuration file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            config = read_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters to MLflow
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Load the test data
            test_data = read_data('data/processed/test_data_processed.csv')

            # Load the model and vectorizer
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Transform the test data using the loaded vectorizer
            x_test_vec = vectorizer.transform(test_data['clean_text']).astype(np.float32)
            y_test = test_data['sentiment']

            # Create model signature for MLflow
            input_example = pd.DataFrame(x_test_vec.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(x_test_vec[:5]))

            # Log the model to MLflow
            mlflow.sklearn.log_model(model, 'model', signature = signature, input_example = input_example)

            # Save model info
            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, model_path,  'model_info.json')

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # Evaluate the model and log metrics
            class_report, conf_matrix, acc_score = evaluate_model(model, x_test_vec, y_test)

            # Log accuracy score to MLflow
            mlflow.log_metric("accuracy_score", acc_score)

            # Log classification report metrics to MLflow
            for cls, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{cls}_{metric}", value)

            # Log confusion matrix as an artifact
            log_conf_matrix(conf_matrix, 'Test')

        except Exception as e:
            logger.error(f"Unexpected error during model evaluation: {e}")
            raise


if __name__ == "__main__":
    main()
