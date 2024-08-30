import os
import io
import sys
import keras
import pickle
from PIL import Image
from SA_hate.logger import logging
from SA_hate.constants import *
from SA_hate.exception import CustomException
from keras.utils import pad_sequences
from SA_hate.configuration.gcloud_syncer import GCloudSync
from SA_hate.components.data_transformation import DataTransformation
from SA_hate.entity.config_entity import DataTransformationConfig
from SA_hate.entity.artifact_entity import DataIngestionArtifacts
import shutil


class PredictionPipeline:
    def __init__(self):
        #self.bucket_name = BUCKET_NAME
        self.storage = GCloudSync()
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.best_model_path = os.path.join(PROJECT_NAME, BEST_MODEL_DIR)
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig,data_ingestion_artifacts=DataIngestionArtifacts)


    
    def get_model_from_storage(self) -> str:
        """
        Method Name :   get_model_from_gcloud
        Description :   This method to get best model from storage
        Output      :   best_model_path
        """
        logging.info("Entered the get_model_from_storage method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket
            os.makedirs(self.model_path, exist_ok=True)
            #self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            #best_model_path = os.path.join(self.model_path, self.model_name)
            
            best_model_path = os.path.join(os.getcwd(), BEST_MODEL_PATH, MODEL_NAME)
            
            #if os.path.isfile(best_model_path) is False:
            logging.info("Check is best model present in the storage or not ?")
            if os.path.isfile(best_model_path) is False:
                print(f"The best model does not exist yet. Please, train the model before.")
                logging.info("The storage model is false and it's necessary to train the model before.")
            else:
                logging.info("Fetch the best model from storage.")
                return best_model_path

            logging.info("Exited the get_model_from_storage method of PredictionPipeline class")

        except Exception as e:
            raise CustomException(e, sys) from e
        

    
    def predict(self, best_model_path, text):
        """load image, returns cuda tensor"""
        logging.info("Running the predict function")
        try:
            #best_model_path: str = self.get_model_from_storage()
            load_model = keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]            
            print(text)
            
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            print(seq)
            
            pred = load_model.predict(padded)
            pred
            print("pred", pred)
            if pred > 0.5:
                print("hate and abusive")
                return "hate and abusive"
            else:
                print("no hate")
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys) from e

    
    def run_pipeline(self, text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            best_model_path: str = self.get_model_from_storage() 
            predicted_text = self.predict(best_model_path, text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e