# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split as tts
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    imp = Imputer(missing_values = float('NaN'), strategy='mean')
    data['LoanAmount'] = imp.fit_transform(data[['LoanAmount']]).ravel()

    data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
    data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
    data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
    data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
    data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
    
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    
    X_train, X_test, y_train, t_test = tts(X, y, test_size = 0.25, random_state = 9)
    
    return X, y, X_train, X_test, y_train, t_test
    
data_cleaning(loan_data)

