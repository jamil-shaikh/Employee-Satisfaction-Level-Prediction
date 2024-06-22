import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load

# Verify the file path
file_path = 'HR_comma_sep.csv'
if not os.path.isfile(HR):
    raise FileNotFoundError(f"The file {HR_comma_sep.csv} does not exist.")

# Load the model
model_2 = load('model_2.pkl')
