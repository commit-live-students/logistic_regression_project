# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    #Impute the values with mean and mode 
    loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean(), inplace = True)
    cat_col = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for col in cat_col:
        loan_data['LoanAmount'].fillna(loan_data[col].mode(), inplace = True)
    
    #seperate the features and target variable
    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]
    
    #train test split for ML
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 9)
    return X, y, X_train, X_test, y_train, y_test







