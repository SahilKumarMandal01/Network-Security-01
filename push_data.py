import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging  

# Load environment variables
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
CA_FILE = certifi.where()

class NetworkDataExtractor:
    """Handles CSV → JSON conversion and MongoDB data insertion."""

    def __init__(self, mongo_url=MONGO_DB_URL, ca_file=CA_FILE):
        try:
            if not mongo_url:
                raise ValueError("MONGO_DB_URL is not set in environment variables.")
            self.mongo_client = pymongo.MongoClient(mongo_url, tlsCAFile=ca_file)
            logging.info("MongoDB client initialized successfully.")
        except Exception as e:
            logging.error("Failed to initialize MongoDB client.")
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, file_path: str):
        """Reads a CSV file and converts it into JSON records."""
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = data.to_dict(orient="records")
            logging.info(f"CSV file {file_path} converted into {len(records)} JSON records.")
            return records
        except Exception as e:
            logging.error(f"Error while converting CSV file {file_path} to JSON.")
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database: str, collection: str):
        """Inserts JSON records into MongoDB collection."""
        try:
            db = self.mongo_client[database]
            coll = db[collection]
            result = coll.insert_many(records)
            logging.info(f"Inserted {len(result.inserted_ids)} records into {database}.{collection}.")
            return len(result.inserted_ids)
        except Exception as e:
            logging.error(f"Error while inserting records into {database}.{collection}.")
            raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    try:
        FILE_PATH = "NetworkData/phisingData.csv"
        DATABASE = "SahilDatabase"
        COLLECTION = "NetworkData"

        extractor = NetworkDataExtractor()
        records = extractor.csv_to_json_converter(FILE_PATH)
        print(records[:2])  # Print only first 2 records for sanity check

        no_of_records = extractor.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Inserted {no_of_records} records successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")