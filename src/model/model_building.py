
# Libraries
import os
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import dagshub
from mlflow.models import infer_signature
import json


# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')


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


# Function to load preprocessed data
def read_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True)
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Unexpected error while reading data: {e}")
        raise


# Function to get the root directory of the project
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


# Function to apply Bag of Words transformation
def appy_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple, models_dir: str):
    try:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        x_train = train_data['clean_text']
        y_train = train_data['sentiment']

        # Fit and transform the training data
        x_train_vec = vectorizer.fit_transform(x_train).astype(np.float32)

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Save vectorizer under /models
        vectorizer_path = os.path.join(models_dir, 'bow_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as file:
            pickle.dump(vectorizer, file)

        logger.debug(f"Bag of Words transformation completed. Train shape: {x_train_vec.shape}")
        return x_train_vec, y_train, vectorizer
    
    except Exception as e:
        logger.error(f"Error during Bag of Words transformation: {e}")
        raise


# Function to train the LightGBM model
def train_model(x_train: np.ndarray, y_train: np.ndarray,
                learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    try:
        best_model = lgb.LGBMClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )

        best_model.fit(x_train, y_train)
        logger.debug("LightGBM model trained successfully.")

        return best_model
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


# Function to save the trained model
def save_model(model: lgb.LGBMClassifier, output_path: str) -> None:
    try:
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise


# Function to save model run info to JSON
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and full artifact path to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"Model info saved to {file_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving the model info: {e}")
        raise


# Main function
def main():
    try:
        # Load configuration parameters
        root_dir = get_root_directory()
        config = read_params(os.path.join(root_dir, 'params.yaml'))

        # Extract model building parameters
        max_features = config['model_building']['max_features']
        ngram_range = tuple(config['model_building']['ngram_range'])
        learning_rate = config['model_building']['learning_rate']
        max_depth = config['model_building']['max_depth']
        n_estimators = config['model_building']['n_estimators']

        # Ensure /models directory exists
        models_dir = os.path.join(root_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Load and preprocess training data
        train_data = read_data(os.path.join(root_dir, 'data/processed/train_data_processed.csv'))
        x_train_vec, y_train, vectorizer = appy_bow(train_data, max_features, ngram_range, models_dir)
        best_model = train_model(x_train_vec, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model under /models
        model_path = os.path.join(models_dir, 'lgbm_model.pkl')
        save_model(best_model, model_path)

        dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-sentiment-analysis-project', mlflow=True)
        mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow')
        mlflow.set_experiment('DVC Pipeline - Model Training')

        with mlflow.start_run(run_name="Model Training") as run:
            input_example = pd.DataFrame(
                x_train_vec.toarray()[:5],
                columns = vectorizer.get_feature_names_out()
            )
            signature = infer_signature(input_example, best_model.predict(x_train_vec[:5]))

            mlflow.sklearn.log_model(
                sk_model = best_model,
                artifact_path = "lgbm_model",
                signature = signature,
                input_example = input_example
            )

            # Log vectorizer from /models
            mlflow.log_artifact(os.path.join(models_dir, 'bow_vectorizer.pkl'))
            logger.info(f"Model logged to MLflow under run {run.info.run_id}")

            # Save the model run info
            artifact_uri = mlflow.get_artifact_uri()
            model_path = f"{artifact_uri}/lgbm_model"
            save_model_info(run.info.run_id, model_path, os.path.join(root_dir, "model_run_info.json"))
            logger.info("Training run info saved to model_info.json with full artifact URI")

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"An error occurred during model building: {e}")


if __name__ == "__main__":
    main()
