import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
# """DECORATOR DECLARED BELOW"""
@dataclass
class Data_Ingestion_CONFIG():
    """CREATING A NEW STRING VARIABLE THAT ENCOMPASSES THE PATH & FOLDER THE RESPECTIVE(TRAIN,TEST,RAW)DATA THAT MUST BE PLACED INTO(HERE INTO THE ARTIFACT FOLDER)."""
    train_path:str= os.path.join("Artifact","train.csv")
    test_path:str= os.path.join("Artifact","test.csv")
    raw_path:str= os.path.join("Artifact","data.csv")
    
class Data_Ingestion():
    def __init__(self):
        self.Ingestion_config=Data_Ingestion_CONFIG()
        """the above object takes form of container to stoe the above 3 paths to read the data from it."""
    
    def Initiate_data_ingestion(self):
        """The primary utility of this function is to:
         READ THE DATA FROM DATABASE """
        logging.info("enter the data ingestion method from the source-(can be obtained from either API SOURCE, DATABASE SOURCE OR MONGODB SOURCE ) ")
        
        try:
            df1=pd.read_csv("notebook\data\stud.csv")
            logging.info("Extracted data from source into the dataframe")

            """Next step is to  CREATE DIRECTORY OR FOLDER FOR EACH OF THE DATAPATHS URL STRING CEATED ABOVE for (TRAIN,TEST, RAW)"""
            os.makedirs(os.path.dirname(self.Ingestion_config.train_path),exist_ok=True)
            df1.to_csv(self.Ingestion_config.raw_path,index=False,header=True)

            logging.info("Train_test split initiatION")

            train_data,test_data=train_test_split(df1,test_size=0.2,random_state=42)
            """Now convert he above dataframes to csv files for both train & test"""
            train_data.to_csv(self.Ingestion_config.train_path,index=False,header=True)
            test_data.to_csv(self.Ingestion_config.test_path,index=False,header=True)

            logging.info("INGESTION COMPLETED")

            return(self.Ingestion_config.train_path,self.Ingestion_config.test_path)
                       
        except Exception as e:
            raise CustomException(e,sys)


if (__name__=="__main__"):
    obj1=Data_Ingestion()
    obj1.Initiate_data_ingestion()

        


