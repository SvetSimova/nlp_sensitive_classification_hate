import os
from datetime import datetime

# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
#BUCKET_NAME = 'hate_speech'
GIT_REPO_URL = 'https://github.com/SvetSimova/nlp_sensitive_classification_hate'
GIT_BRANCH = "main"  # Specify the branch, e.g., 'main', 'master', or any other branch name
GIT_ZIPFILE_PATH = "data/download/dataset.zip"  # Path to the file in the repo
GIT_RAW_URL_DATA = f"https://raw.githubusercontent.com/{GIT_REPO_URL.split('/')[-2]}/{GIT_REPO_URL.split('/')[-1]}/{GIT_BRANCH}/{GIT_ZIPFILE_PATH}"
ZIP_FILE_NAME = 'dataset.zip'
LABEL = "label"
TWEET = "tweet"

# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"


# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
TRANSFORMED_FILE_NAME = "final.csv"
DATA_DIR = "data"
ID = 'id'
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['count', 'hate_speech', 'offensive_language', 'neither']
CLASS = 'class'


# Model training constants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.h5'
X_TEST_FILE_NAME = 'X_test.csv'
Y_TEST_FILE_NAME = 'y_test.csv'

X_TRAIN_FILE_NAME = 'X_train.csv'

RANDOM_STATE = 42
EPOCH = 1
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# Model Architecture constants
MAX_WORDS = 50000
MAX_LEN = 300
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
ACTIVATION = 'sigmoid'


# Model  Evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = 'ModelEvaluationArtifacts'
BEST_MODEL_DIR = "best_model"
MODEL_EVALUATION_FILE_NAME = 'loss.csv'
MODEL_NAME = 'model.h5'
GIT_RAW_URL_MODEL = f"https://raw.githubusercontent.com/{GIT_REPO_URL.split('/')[-2]}/{GIT_REPO_URL.split('/')[-1]}/{GIT_BRANCH}/{BEST_MODEL_DIR}/{MODEL_NAME}"


APP_HOST = "0.0.0.0"
APP_PORT = 8080