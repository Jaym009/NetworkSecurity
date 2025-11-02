from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## Configuration for data ingestion config
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self) -> List[pymongo.collection.Collection]:
        try:
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            database = mongo_client[self.data_ingestion_config.database_name]
            collection = database[self.data_ingestion_config.collection_name]
            data = list(collection.find())
            dataframe = pd.DataFrame(data)
            if "_id" in dataframe.columns:
                dataframe = dataframe.drop(columns=["_id"], axis=1)
            dataframe.replace({"na":np.nan}, inplace=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> str:
        try:
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            return self.data_ingestion_config.feature_store_file_path
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
         
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
