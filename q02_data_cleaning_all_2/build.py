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


# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    X_train['LoanAmount'] = np.sqrt(X_train['LoanAmount'])
    X_train['ApplicantIncome'] = np.sqrt(X_train['ApplicantIncome'])
    X_train['CoapplicantIncome'] = np.sqrt(X_train['CoapplicantIncome'])

    X_test['LoanAmount'] = np.sqrt(X_test['LoanAmount'])
    X_test['ApplicantIncome'] = np.sqrt(X_test['ApplicantIncome'])
    X_test['CoapplicantIncome'] = np.sqrt(X_test['CoapplicantIncome']) 

    get_dummies_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    X_train = pd.get_dummies(X_train, columns = get_dummies_cols, drop_first = True)
    X_test = pd.get_dummies(X_test, columns = get_dummies_cols, drop_first = True)
    
    return X_train, X_test, y_train, y_test
    
data_cleaning_2(X_train, X_test, y_train, y_test)


