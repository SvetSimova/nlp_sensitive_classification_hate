from SA_hate.logger import logging
from SA_hate.exception import CustomException
import sys
#from SA_hate.configuration.gcloud_syncer import DataSync
import requests
import os
import subprocess
#from SA_hate.components.model_evaluation import ModelEvaluation
from SA_hate.entity.config_entity import ModelEvaluationConfig
from SA_hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts
from SA_hate.constants import *
from pathlib import Path
import shutil

#logging.info("Welcome to our Project")

# try:
#     a = 7 / "0"

# except Exception as e:
#     raise CustomException(e, sys)

# obj = GCloudSync()
# obj.sync_folder_from_gcloud(bucket_name, "dataset.zip", "download/dataset.zip")

# obj = DataSync()
# data_url = 'https://github.com/SvetSimova/nlp_sensitive_classification_hate/blob/main/data/download/dataset.zip'
# obj.sync_folder_from_gitrepo(data_url, "dataset.zip", "download/dataset.zip")

# self.GIT_MODEL_URL = GIT_RAW_URL_MODEL
# self.MODEL_EVALUATION_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
# self.BEST_MODEL_DIR_PATH: str = os.path.join(self.MODEL_EVALUATION_MODEL_DIR, BEST_MODEL_DIR)
# self.MODEL_NAME = MODEL_NAME

# MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
# BEST_MODEL_DIR = "best_model"
# MODEL_EVALUATION_FILE_NAME = 'loss.csv'
#MODEL_NAME = 'model.h5'
# GIT_RAW_URL_MODEL = f"https://raw.githubusercontent.com/{GIT_REPO_URL.split('/')[-2]}/{GIT_REPO_URL.split('/')[-1]}/{GIT_BRANCH}/{BEST_MODEL_DIR}/{MODEL_NAME}"

#obj = ModelPush(model_evaluation_config=self.model_evaluation_config, model_trainer_artifacts=model_trainer_artifacts)
#os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)
                # artifacts/08_26_2024_23_07_56/ModelEvaluationArtifacts/best_model
#source = os.path.join(os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR), 
                                  #MODEL_NAME)
try:
    is_model_accepted = False
    best_model_path = os.path.join(os.getcwd(), BEST_MODEL_PATH, MODEL_NAME)
    if os.path.isfile(best_model_path):
        print(f"The best model exists !")
    else:
        print(f"The best model does not exist yet.")
    if os.path.isfile(best_model_path) is False:
        is_model_accepted = True
    print(is_model_accepted)
except Exception as e:
    raise CustomException(e, sys) from e


# Define repository details
repo_url = "https://github.com/SvetSimova/nlp_sensitive_classification_hate.git"
local_repo_path = "/path/to/local/repo"  # Path where you want to clone the repo
model_path = "/path/to/model.h5"  # Path to the model.h5 file
commit_message = "Add model.h5 file"


#os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

# response = requests.get(self.model_evaluation_config.GIT_MODEL_URL)
# # Copy the model.h5 file to the repository directory
# destination_path = os.path.join(local_repo_path, "model.h5")
# shutil.copy(model_path, destination_path)

# # Change the current working directory to the local repository path
# os.chdir(local_repo_path)

# # Stage the model.h5 file
# subprocess.run(["git", "add", "model.h5"])

# # Commit the changes
# subprocess.run(["git", "commit", "-m", commit_message])

# # Push the changes to the remote repository
# subprocess.run(["git", "push", "origin", "main"])