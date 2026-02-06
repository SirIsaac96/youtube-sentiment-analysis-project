
# Libraries
import os
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer


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


# Model building functions
def read_params(config_path: str) -> dict:
    """Reads the YAML configuration file."""
    try:
        # Read the YAML configuration file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.debug(f"Configuration loaded from {config_path}")
        return config
    
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def read_data(file_path: str) -> pd.DataFrame:
    """Reads the CSV data file into a DataFrame."""
    try:
        # Read the data from the specified file path
        data = pd.read_csv(file_path)
        data.fillna('', inplace=True) # Fill any NaN values with empty strings
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while reading data: {e}")
        raise


def get_root_directory() -> str:
    """Get the root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def appy_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply Bag of Words vectorization to the text data."""
    try:
        # Initialize CountVectorizer 
        vectorizer = CountVectorizer(max_features = max_features, ngram_range = ngram_range)

        # Feature engineering
        x_train = train_data['clean_text']
        y_train = train_data['sentiment']

        # BoW Vectorization
        x_train_vec = vectorizer.fit_transform(x_train).astype(np.float32)
        logger.debug(f"Bag of Words transformation completed successfully. Train shape: {x_train_vec.shape}")

        # Save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(),'bow_vectorizer.pkl'), 'wb') as file:
            pickle.dump(vectorizer, file)

        logger.debug("Bag of Words transformation applied with trigrams and data transformed.")

        return x_train_vec, y_train
    
    except Exception as e:
        logger.error(f"Error during Bag of Words transformation: {e}")
        raise


def train_model(x_train: np.ndarray, y_train: np.ndarray, 
                learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        # Define the best model with the specified hyperparameters
        best_model = lgb.LGBMClassifier(
            learning_rate = learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators,
            objective = 'multiclass',
            num_class = 3,
            metric = 'multi_logloss',
            class_weight = 'balanced',
            reg_alpha = 0.1,
            reg_lambda = 0.1,
            random_state = 42
        )

        # Train the model
        best_model.fit(x_train, y_train)
        logger.debug("LightGBM model trained successfully.")

        return best_model
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model: lgb.LGBMClassifier, output_path: str) -> None:
    """Saves the trained model to a file."""
    try:
        # Save the model to the specified output path
        with open(output_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"Model saved successfully to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise


def main():
    try:
        # Get the root directory
        root_dir = get_root_directory()
        logger.debug(f"Root directory determined: {root_dir}")

        # Read configuration parameters
        config = read_params(os.path.join(root_dir, 'params.yaml'))
        max_features = config['model_building']['max_features']
        ngram_range = tuple(config['model_building']['ngram_range'])
        learning_rate = config['model_building']['learning_rate']
        max_depth = config['model_building']['max_depth']
        n_estimators = config['model_building']['n_estimators']

        # Read the processed data
        train_data = read_data(os.path.join(root_dir, 'data/processed/train_data_processed.csv'))

        # Apply Bag of Words vectorization
        x_train_vec, y_train = appy_bow(train_data, max_features, ngram_range)

        # Train the model
        best_model = train_model(x_train_vec, y_train, learning_rate, max_depth, n_estimators)

        # Save the model
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"An error occurred during model building: {e}")


if __name__ == "__main__":
    main()
