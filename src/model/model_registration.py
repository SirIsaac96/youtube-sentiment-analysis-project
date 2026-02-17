
# Libraries
import json
import mlflow
import dagshub
import logging
from mlflow.tracking import MlflowClient
from requests.exceptions import ConnectionError


# Logging configuration
logger = logging.getLogger('model_registration')
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


# MLflow setup with DagsHub
dagshub.init(
    repo_owner='SirIsaac96',
    repo_name='youtube-sentiment-analysis-project',
    mlflow=True
)
tracking_uri = 'https://dagshub.com/SirIsaac96/youtube-sentiment-analysis-project.mlflow'
mlflow.set_tracking_uri(tracking_uri)


# Function to load experiment information from JSON file
def load_experiment_info(file_path: str) -> dict:
    """Load experiment information from JSON file."""
    try:
        with open(file_path, 'r') as file:
            info = json.load(file)
        logger.debug(f"Experiment info loaded from {file_path}: {info}")
        return info
    
    except FileNotFoundError:
        logger.error("Experiment info file not found.")
        raise

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from experiment info file: {e}")
        raise


# Model registration function
def register_model(model_name: str, experiment_info: dict):
    """Register the best model using the full artifact URI."""
    try:
        client = MlflowClient()

        # Experiment info
        model_uri = experiment_info['model_path']
        logger.debug(f"Registering model from URI: {model_uri}")

        # Create registered model if not exists
        try:
            client.create_registered_model(model_name)
            logger.debug(f"Registered model '{model_name}' created successfully.")

        except Exception:
            logger.debug(f"Registered model '{model_name}' already exists.")

        # Register model version
        model_version = mlflow.register_model(
            model_uri = model_uri,
            name = model_name
        )
        logger.info(f"Model version {model_version.version} registered successfully under name: {model_name}")

        # Assign alias and transition stage
        client.set_registered_model_alias(
            name = model_name,
            alias = 'Best_Model',
            version = model_version.version
        )
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "Production"
        )
        logger.info("Alias 'Best_Model' assigned and model transitioned to Production successfully.")

    except ConnectionError as e:
        logger.error(f"Error connecting to MLflow tracking server at {tracking_uri}: {e}")
        print("Failed to connect to MLflow tracking server. Please check if it's running and accessible.")
        raise

    except Exception as e:
        logger.error(f"Unexpected error while registering model: {e}")
        print(f"An unexpected error occurred during model registration: {e}")
        raise


# Main function
def main():
    try:
        # Load info from model_run_info.json (created during training)
        experiment_info = load_experiment_info('model_run_info.json')
        register_model('YouTube_Sentiment_Analysis_Model', experiment_info)

    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        print(f"An error occurred during model registration: {e}")
        raise


if __name__ == "__main__":
    main()
