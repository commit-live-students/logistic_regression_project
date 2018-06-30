# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


def data_cleaning(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    np.random.seed=9
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    mean_imputer = Imputer(strategy='mean')
    mode_imputer = Imputer(strategy='most_frequent')
    for i in range(len(X_train.columns)):
        col = X_train.columns[i]
        if X_train.iloc[:,i].dtype != np.object:
            X_train.iloc[:,i] = mean_imputer.fit_transform(X_train[[col]])
        else:
            X_train.iloc[:,i].fillna(X_train.iloc[:,i].mode(), inplace=True)
    for i in range(len(X_test.columns)):
        col = X_test.columns[i]
        if X_test.iloc[:,i].dtype != np.object:
            X_test.iloc[:,i] = mean_imputer.fit_transform(X_test[[col]])
        else:
            X_test.iloc[:,i].fillna(X_test.iloc[:,i].mode(), inplace=True)
    return X, y, X_train, X_test, y_train, y_test


