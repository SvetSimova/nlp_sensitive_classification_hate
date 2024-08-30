import sys
from SA_hate.logger import logging
from SA_hate.exception import CustomException
from SA_hate.entity.config_entity import ModelPusherConfig
from SA_hate.entity.artifact_entity import ModelPusherArtifacts
import os
from SA_hate.constants import *
import shutil

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        Params: model_pusher_config - Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
    
    
    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")
        try:
            # Uploading the model to storage
            source = os.path.join(self.model_pusher_config.TRAINED_MODEL_PATH, 
                                  self.model_pusher_config.MODEL_NAME)
            destination = os.path.join(BEST_MODEL_PATH, MODEL_NAME)

            shutil.copy(source, destination)
            logging.info("Record the best model to the internal folder")

            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifacts(
                best_model=self.model_pusher_config.MODEL_NAME
            )
            
            logging.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact
        

        except Exception as e:
            raise CustomException(e, sys) from e
