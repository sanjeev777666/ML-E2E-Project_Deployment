import os
import sys
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass

class DataTxnConfig():
    """CREATING A NEW STRING VARIABLE THAT ENCOMPASSES THE PATH & FOLDER THE RESPECTIVE(TRAIN,TEST,RAW)DATA THAT MUST BE PLACED INTO(HERE INTO THE ARTIFACT FOLDER)."""
    preprocessor_obj_file_path= os.path.join("Artifact","preprocessor.pkl")
class DataTxn():
    def __init__(self):
        self.Txn_config=DataTxnConfig()
        """the above object takes form of container to store the the path url which is to be txn'd to pkl fil using dill(alternative to pickle method)."""
    
    
    
    # Current data txn is used to perform  EDA  & FE  
    def get_data_txn_obj(self):

        """This function si responsible for data trnasformation"""
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            """FOLLOWING SECTION CODE REPSONISBLE FOR HANDLING MISSING VALUES& ENCODING"""    
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                
                ]
            )
            
            #"""FOLLOWING SECTION HANDLES SCALING &, IMPUTING NUMERICAL COLUMNS & CATEGORICAL TO NUMERICAL/ ONE HOT ENCODING"""
            cat_pipeline=Pipeline(

                    steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]

            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}") 

            preprocessor=ColumnTransformer(
                    [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)

                    ]


            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_txn(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("OBTAINING PREPROCESSING_OBJECT")

            preprocessing_obj=self.get_data_txn_obj()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe ABOVE ."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            logging.info(f"NOW Save THE ABOVE preprocessing object.BELOW")

            save_object(file_path=self.Txn_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            
            return(
                train_arr,
                test_arr,
                # self.Txn_config.preprocessor_obj_file_path
            )
               
        except Exception as e:
            raise CustomException(e,sys)
