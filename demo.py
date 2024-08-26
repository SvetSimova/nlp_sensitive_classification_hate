from SA_hate.logger import logging
from SA_hate.exception import CustomException
import sys
from SA_hate.configuration.gcloud_syncer import DataSync
import requests

#logging.info("Welcome to our Project")

# try:
#     a = 7 / "0"

# except Exception as e:
#     raise CustomException(e, sys)

# obj = GCloudSync()
# obj.sync_folder_from_gcloud(bucket_name, "dataset.zip", "download/dataset.zip")

obj = DataSync()
data_url = 'https://github.com/SvetSimova/nlp_sensitive_classification_hate/blob/main/data/download/dataset.zip'
obj.sync_folder_from_gitrepo(data_url, "dataset.zip", "download/dataset.zip")