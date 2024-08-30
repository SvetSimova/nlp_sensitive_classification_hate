import os
import sys
import pickle
import pandas as pd
from SA_hate.logger import logging
from SA_hate.exception import CustomException
from keras.models import load_model
from keras.utils import pad_sequences
from SA_hate.constants import *
from sklearn.metrics import confusion_matrix
from SA_hate.entity.config_entity import ModelEvaluationConfig
from SA_hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts
#from SA_hate.configuration.gcloud_syncer import GCloudSync


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        Params: 
        model_evaluation_config: Configuration for model evaluation
        data_transformation_artifacts: Data Transformation artifact stage
        model_trainer_artifacts: Output reference of model trainer artifact stage
        """

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        #self.gcloud = GCloudSync()

    
    def get_best_model_from_storage(self) -> str:
        """
        Return: Fetch the best model from storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_storage method of Model Evaluation class")
            
            # BEST_MODEL_DIR_PATH = os.path.join(self.MODEL_EVALUATION_MODEL_DIR, BEST_MODEL_DIR)
            # MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
            # result: BEST_MODEL_DIR_PATH = "artifacts/timestamp/ModelEvaluationArtifacts/best_model"
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            # from {PROJECT_NAME}/best_model/model.h5
            best_model_path = os.path.join(os.getcwd(), BEST_MODEL_PATH, MODEL_NAME)
            
            #if os.path.isfile(best_model_path) is False:
            if os.path.isfile(best_model_path):
                logging.info("The best model exists!")
                print(f"The best model exists !")
            else:
                logging.info("The best model does not exist yet.")
                print(f"The best model does not exist yet.")
                
            logging.info("Exited the get_best_model_from_storage method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e 
        

    def our_evaluate(self, model, tokenizer):
        """
        Params:
        model: Currently trained model or best model from the storage
        data_loader: Data loader for validation dataset
        
        Return: loss
        """
        try:
            logging.info("Entering into to the evaluate function of Model Evaluation class")
            print(self.model_trainer_artifacts.X_test_path)
            X_test = pd.read_csv(self.model_trainer_artifacts.X_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            # with open('tokenizer.pickle', 'rb') as handle:
            #     tokenizer = pickle.load(handle)

            # load_our_model = load_model(self.model_trainer_artifacts.trained_model_path)

            X_test = X_test['tweet'].astype(str)
            X_test = X_test.squeeze()
            y_test = y_test.squeeze()
            print(f"-----------------{X_test.shape}--------------")
            print(f"-----------------{y_test.shape}--------------")

            test_sequences = tokenizer.texts_to_sequences(X_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            accuracy = model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"Accuracy is {accuracy}")

            lstm_prediction = model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)
 
            print(confusion_matrix(y_test, res))
            print(f"Accuracy is {accuracy}")
            logging.info(f"Confusion_matrix is: {confusion_matrix(y_test, res)} ")
            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e
 
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
            Method Name :   initiate_model_evaluation
            Description :   This function is used to initiate all steps of the model evaluation

            Output      :   Returns model evaluation artifact
            On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Initiate Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            #trained_model = load_model(self.model_trainer_artifacts.trained_model_path)
            trained_model = load_model(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            trained_model_accuracy = self.our_evaluate(trained_model, load_tokenizer)
            print(f"The trained_model accuracy is {trained_model_accuracy}")

            logging.info("Fetch the best model from the storage")
            best_model_path = self.get_best_model_from_storage()

            logging.info("Check is best model present in the storage or not ?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("The storage model is false and currently trained model accepted is true")
            else:
                logging.info("Load best model fetched from the storage")
                best_model = load_model(best_model_path)
                best_model_accuracy = self.our_evaluate(best_model, load_tokenizer)
                print(f"The best_model accuracy is {best_model_accuracy}")

                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_accuracy[1] > trained_model_accuracy[1]:
                    is_model_accepted = False
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = True
                    logging.info("Trained model accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
