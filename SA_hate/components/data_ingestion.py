import os
import io
import sys
import requests
from zipfile import ZipFile
from SA_hate.logger import logging
from SA_hate.exception import CustomException
from SA_hate.entity.config_entity import DataIngestionConfig
from SA_hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
    #     self.ZIP_FILE_NAME = ZIP_FILE_NAME
    #     self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
    #     self.DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
    #     self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_RAW_DATA_DIR)
    #     self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
    #     self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)

# ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
# BUCKET_NAME = 'hate_speech'
# GIT_REPO_URL = 'https://github.com/SvetSimova/nlp_sensitive_classification_hate'
# GIT_BRANCH = "main"  # Specify the branch, e.g., 'main', 'master', or any other branch name
# GIT_ZIPFILE_PATH = "data/download/dataset.zip"  # Path to the file in the repo
# GIT_RAW_URL = f"https://raw.githubusercontent.com/{github_repo_url.split('/')[-2]}/{github_repo_url.split('/')[-1]}/{branch}/{filename}"
# ZIP_FILE_NAME = 'dataset.zip'
# LABEL = "label"
# TWEET = "tweet"

# # Data ingestion constants
# DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
# DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
# DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"

    def get_data(self) -> None:
        logging.info("Start get_data method of DataIngestion class")
        try:
            response = requests.get(self.data_ingestion_config.GIT_DATA_URL)
            # Check if the request was successful
            if response.status_code == 200:
                # Create the destination directory if it doesn't exist
                os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
                    #os.path.join(os.getcwd(), "artifacts", TIMESTAMP, "DataIngestionArtifacts")
                
                with ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR)

                print(f"File downloaded and unzipped successfully to {self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR}")

                return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

            logging.info("Exited the get_data method of DataIngestion class")

        
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")

        try:
            logging.info("Fetch the data from GitHub repository")
            imbalance_data_file_path, raw_data_file_path = self.get_data()

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path= imbalance_data_file_path,
                raw_data_file_path = raw_data_file_path
            )

            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e       
    