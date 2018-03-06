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
    #X_train1 = X_train.select_dtypes(include=['float64', 'int64'])
    cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    for col in cols:
        X_train[col] = np.sqrt(X_train[col])
        X_test[col] = np.sqrt(X_test[col])

    col1 = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    #for col_1 in col1:
    X_train = pd.get_dummies(X_train, columns=col1, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=col1, drop_first=True)
    return X_train, X_test, y_train, y_test
print data_cleaning_2(X_train, X_test, y_train, y_test)
