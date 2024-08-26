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
    