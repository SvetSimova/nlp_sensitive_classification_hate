import sys
from SA_hate.logger import logging
from SA_hate.exception import CustomException

from SA_hate.components.data_ingestion import DataIngestion
from SA_hate.entity.config_entity import (DataIngestionConfig)
from SA_hate.entity.artifact_entity import (DataIngestionArtifacts)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from GitHub repository")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from GitHub repository")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            # data_transformation_artifacts = self.start_data_transformation(
            #     data_ingestion_artifacts=data_ingestion_artifacts
            # )

            # model_trainer_artifacts = self.start_model_trainer(
            #     data_transformation_artifacts=data_transformation_artifacts
            # )

            # model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts,
            #                                                         data_transformation_artifacts=data_transformation_artifacts
            # ) 

            # if not model_evaluation_artifacts.is_model_accepted:
            #     raise Exception("Trained model is not better than the best model")
            
            # model_pusher_artifacts = self.start_model_pusher()
            logging.info("Exited the run_pipeline method of TrainPipeline class") 

        except Exception as e:
            raise CustomException(e, sys) from e
        
        

