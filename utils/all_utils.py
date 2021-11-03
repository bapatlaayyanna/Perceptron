import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

def prepare_data(df):
  """It used to separate dependent and independent data

  Args:
      df (Pandas DataFrame): Its data set

  Returns:
      tuples: it returns the tuples of dependent and independent variables
  """
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y

def save_model(model, filename):
  """This method is to save model using joblib

  Args:
      model (python object): tarined model    
      filename (str): path to save model  
  """
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) 
  filePath = os.path.join(model_dir, filename) 
  joblib.dump(model, filePath)