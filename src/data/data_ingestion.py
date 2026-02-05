
# Libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging


# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


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
        logger.debug(f"Data loaded from {file_path} with shape {data.shape}")
        return data
    
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data by handling missing values, duplicates and empty strings."""
    try:
        data = data.dropna() # drop missing values
        data = data.drop_duplicates(inplace = True) # drop duplicates
        data = data[~(data['clean_text'].str.strip() == '')] # drop empty strings
        logger.debug(f"Dropped missing values, duplicates and empty strings. Remaining data shape: {data.shape}")
        return data
    
    except KeyError as e:
        logger.error(f"Column 'clean_text' not found in the dataframe: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during data preprocessing: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    """Saves the train and test sets to a CSV file."""
    try:
        # Define the path for the raw data
        raw_data_path = os.path.join(output_path, 'raw')

        # Create the directory (data/raw) if it doesn't exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test sets to the raw data directory
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
        logger.debug(f"Train and test data saved to {raw_data_path}")

    except Exception as e:
        logger.error(f"Error saving train and test data to {output_path}: {e}")
        raise


def main():
    try:
        # Read configuration parameters
        config = read_params(config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = config['data_ingestion']['test_size']

        # Read the data
        data = read_data('data/interim/youtube_sentiments.csv')

        # Preprocess the data
        data = preprocess_data(data)

        # Split the data into train and test sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        logger.debug(f"Data split into train and test sets with test size {test_size}")

        # Save the train and test sets
        save_data(train_data, test_data, data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

    except Exception as e:
        logger.error(f"Failed to complete data ingestion process: {e}")
        raise


if __name__ == "__main__":
    main()
