import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.exception import CustomException 
import dill # help to create pickel file


def save_object(file_path, object):
    """
    Save the object into the given file path
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    