# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
import matplotlib.pyplot as plt


loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
# Write your solution here :

def data_cleaning_2(X_train, X_test, y_train, y_test):
    num_columns = X_train._get_numeric_data().columns
    cat_columns = set(X_train.columns) - set(X_train._get_numeric_data().columns)

    for name in num_columns:
        X_train[name] = np.sqrt(X_train[name])
        X_test[name] = np.sqrt(X_test[name])

    for name in cat_columns:
        dummy_train = pd.get_dummies(X_train[name],prefix=name,drop_first=True)
        X_train = pd.concat([X_train,dummy_train],axis=1)
        X_train.drop(name, axis=1, inplace=True)
        
        dummy_test = pd.get_dummies(X_test[name],prefix=name,drop_first=True)
        X_test = pd.concat([X_test,dummy_test],axis=1)
        X_test.drop(name, axis=1, inplace=True)

    return X_train, X_test, y_train, y_test

#data_cleaning_2(X_train, X_test, y_train, y_test)

