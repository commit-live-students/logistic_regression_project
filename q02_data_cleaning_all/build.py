# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    X,y = data.iloc[:,:-1],data.iloc[:,-1]
    X['LoanAmount'] = X['LoanAmount'].fillna(X['LoanAmount'].mean())
    cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']

    for col in cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    return X,y,X_train,X_test,y_train,y_test
