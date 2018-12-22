# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

# Write your solution here :
def data_cleaning(data):
    
    
    
    numeric_data = data._get_numeric_data()
    categorical_data = data[list(set(data.columns) - set(numeric_data.columns))]
    
    imputer = Imputer(missing_values=np.nan,strategy='mean',axis=0)
    numeric_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    
    for column in categorical_data.columns:
        categorical_data[column].replace(np.nan, categorical_data[column].mode())
    
    data = pd.concat([numeric_data, categorical_data], 1)
    
    X = data.drop('Loan_Status', 1)
    y = data['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    return X, y, X_train, X_test, y_train, y_test
#     return categorical_data



