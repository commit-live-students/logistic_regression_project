# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

def data_cleaning_2(X_train, X_test, y_train, y_test):
    
    numeric_feature = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Credit_History']
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for feature in numeric_feature:
        X_train[feature] = np.sqrt(X_train[feature])
        X_test[feature] = np.sqrt(X_test[feature])
    
    X_train_dummy = pd.get_dummies(X_train[cat_features], drop_first=True)
    X_test_dummy = pd.get_dummies(X_test[cat_features], drop_first=True)
    X_train = X_train[numeric_feature].join(X_train_dummy)
    X_test = X_test[numeric_feature].join(X_test_dummy)
    y_train, y_test = y_train, y_test
    
    return X_train, X_test, y_train, y_test

data_cleaning_2(X_train, X_test, y_train, y_test)
   
    


