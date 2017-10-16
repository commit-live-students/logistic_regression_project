# Default Imports
import sys,os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.curdir)))
from q02_data_cleaning_all.build import data_cleaning
from q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :

