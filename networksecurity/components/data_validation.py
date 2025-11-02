from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from pandas.api.types import is_numeric_dtype
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config['columns'])
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Verify that all columns listed under 'numerical_columns' in the schema
        exist in the dataframe and have numeric dtypes.
        """
        try:
            expected = self.schema_config.get('numerical_columns', [])
            # check presence
            missing = [col for col in expected if col not in dataframe.columns]
            if missing:
                logging.info(f"Missing numerical columns: {missing}")
                return False

            # check dtype
            non_numeric = [col for col in expected if not is_numeric_dtype(dataframe[col])]
            if non_numeric:
                logging.info(f"Columns not numeric: {non_numeric}")
                return False

            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    isfound = False
                else:
                    isfound = True
                    status = False
                report.update({column:{"pvalue":float(is_same_dist.pvalue),
                                       "drift_status":isfound}})
                
            ## create directory
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read data from train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # validate number of columns
            train_status = self.validate_number_of_columns(train_df)
            test_status = self.validate_number_of_columns(test_df)
            if not train_status: 
                logging.info(f"Train file: {train_file_path} failed the column number validation")
            if not test_status:
                logging.info(f"Test file: {test_file_path} failed the column number validation")

            # validate numerical columns using schema
            train_num_status = self.validate_numerical_columns(train_df)
            test_num_status = self.validate_numerical_columns(test_df)
            if not train_num_status:
                logging.info(f"Train file: {train_file_path} failed numerical columns validation")
            if not test_num_status:
                logging.info(f"Test file: {test_file_path} failed numerical columns validation")
            
            ## lets check data drift
            drift_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            if not drift_status:
                logging.info(f"Data drift found between train and test dataset")
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status = train_status and test_status and drift_status,
                valid_train_file_path = self.data_validation_config.valid_train_file_path,
                valid_test_file_path = self.data_validation_config.valid_test_file_path,
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e