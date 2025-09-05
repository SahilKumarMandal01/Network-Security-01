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
