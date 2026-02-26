
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


# Utility functions
# Function to load data
def read_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


# Function to load model
def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise


# Function to load CountVectorizer
def load_vectorizer(vectorizer_path: str) -> CountVectorizer:
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug(f"CountVectorizer loaded from {vectorizer_path}")
        return vectorizer
    
    except Exception as e:
        logger.error(f"Unexpected error while loading vectorizer: {e}")
        raise


# Function to read YAML configuration file
def read_params(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Configuration loaded from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        raise


# Function to evaluate model performance
def evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed.")
        return class_report, conf_matrix, acc_score
    
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


# Function to log confusion matrix as an image in MLflow
def log_conf_matrix(conf_matrix, dataset_name):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix for {dataset_name} Dataset')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        mlflow.log_figure(fig, f'conf_matrix_{dataset_name}.png')
        plt.close(fig)
        logger.debug(f"Confusion matrix for {dataset_name} dataset logged successfully.")

    except Exception as e:
        logger.error(f"Error logging confusion matrix: {e}")
        raise


# Function to save model information
def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        with open(file_path, 'w') as json_file:
            json.dump(model_info, json_file, indent=4)
        logger.debug(f"Model information saved to {file_path} successfully.")

    except Exception as e:
        logger.error(f"Error saving model information: {e}")
        raise


# Main function
def main():
    dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-sentiment-analysis-project', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow')
    mlflow.set_experiment('DVC Pipeline - Model Evaluation')

    with mlflow.start_run(run_name='Model Evaluation') as run:
        try:
            mlflow.set_tag("author", "Isaac-Otom")
            mlflow.set_tag("stage", "Evaluation")

            # Load configuration parameters
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            config = read_params(os.path.join(root_dir, 'params.yaml'))

            # Log all configuration parameters
            for key, value in config.items():
                mlflow.log_param(key, value)

            # Ensure /models directory exists
            models_dir = os.path.join(root_dir, 'models')

            # Load test data, model, and vectorizer
            test_data = read_data(os.path.join(root_dir, 'data/processed/test_data_processed.csv'))
            model = load_model(os.path.join(models_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(models_dir, 'bow_vectorizer.pkl'))

            # Transform test data
            x_test_vec = vectorizer.transform(test_data['clean_text']).astype(np.float32)
            y_test = test_data['sentiment']

            # Load model run info
            with open(os.path.join(root_dir, 'model_run_info.json'), 'r') as f:
                training_info = json.load(f)
            model_run_id = training_info['run_id']
            model_path = training_info['model_path'] 

            # Save model info for registration stage
            save_model_info(model_run_id, model_path, os.path.join(root_dir, 'model_info.json'))

            # Evaluate and log metrics
            class_report, conf_matrix, acc_score = evaluate_model(model, x_test_vec, y_test)
            mlflow.log_metric("accuracy_score", acc_score)

            for cls, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        mlflow.log_metric(f"{cls}_{metric}", value)

            log_conf_matrix(conf_matrix, 'Test')

        except Exception as e:
            logger.error(f"Unexpected error during model evaluation: {e}")
            raise


if __name__ == "__main__":
    main()
