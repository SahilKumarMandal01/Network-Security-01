import os
import sys
import yaml
import pickle
import numpy as np
from typing import Any, Dict
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: Any, replace: bool=False) -> None:
    """
    Writes a dictionary or object to a YAML file.
    
    Args: 
        file_path (str): Path to save the YAML fiel.
        content (Any): The content to write.
        replace (bool): If True, replaces the file if it exists.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        
        logging.info(f"YAML file written at: {file_path}")
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saved a Numpy array to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info(f"Numpy array saved at: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a Numpy array from a file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: Any) -> None:
    """
    Serializes and saves a Python object using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Loads a serailized Python object from a file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded from: {file_path}") 
        return obj
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)   