# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(df):
    X = loan_data.iloc[:, :-1]
    y = loan_data['Loan_Status']


    X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].mean())
    #X['LoanAmount'] = X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())

    col_list = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for cols in col_list:
        X[cols] = X[cols].fillna(X[cols].mode()[0])
        #X_test[cols] = X_test[cols].fillna(X_test[cols].mode()[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return ((X)), (y), (X_train), (X_test), (y_train), (y_test)

print data_cleaning(loan_data)
