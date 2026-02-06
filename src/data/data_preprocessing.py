
# Libraries
import os
import re
import numpy as np
import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging


# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)


file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')


# Preprocessing functions
def preprocess_text(text: str) -> str:
    """Apply preprocessing transformations to a text."""
    try:
        # 1. Convert to lowercase
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
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the data by applying preprocessing to the 'text' column."""
    try:
        # Apply preprocessing to the 'text' column
        data['clean_text'] = data['text'].apply(preprocess_text)
        logger.debug(f"Data normalization completed successfully. Data shape after normalization: {data.shape}")
        return data
    
    except Exception as e:
        logger.error(f"Unexpected error during data normalization: {e}")
        raise


def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    """Saves the processed train and test datasets to a CSV file."""
    try:
        # Define the path for the processed data
        processed_data_path = os.path.join(output_path, 'processed')
        logger.debug(f'Creating directory for processed data at: {processed_data_path}')

        # Create the directory (data/processed) if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)
        logger.debug(f'Directory created at: {processed_data_path} or already exists.')

        # Save the DataFrame to a CSV file
        train_data.to_csv(os.path.join(processed_data_path, 'train_data_processed.csv'), index=False)
        test_data.to_csv(os.path.join(processed_data_path, 'test_data_processed.csv'), index=False)
        logger.debug(f"Processed data saved to {processed_data_path}")
    
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def main():
    """Main function to execute data preprocessing."""
    try:
        logger.debug("Starting data preprocessing...")

        # Read the raw data
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug("Raw data loaded successfully.")

        # Preprocess the data
        train_data_processed = normalize_data(train_data)
        test_data_processed = normalize_data(test_data)
        logger.debug("Data normalization completed successfully.")

        # Save the processed data
        save_processed_data(train_data_processed, test_data_processed, output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
        logger.debug("Data preprocessing completed successfully.")

    except Exception as e:
        logger.error(f"Failed to complete data preprocessing process: {e}")
        print(f'Error: {e}')


if __name__ == "__main__":
    main()