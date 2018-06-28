# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer
# from logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    X = data.copy().drop(['Loan_Status'], axis=1)
    y = data.Loan_Status;
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state = 9)
    imp_mean = Imputer(missing_values='NaN', strategy='mean')
    num_cols = X_train._get_numeric_data().columns
    for col in num_cols:
        imp_mean.fit(X_train[[col]])
        X_train.LoanAmount = imp_mean.fit_transform(X_train[[col]])
        imp_mean.fit(X_test[[col]])
        X_test.LoanAmount = imp_mean.fit_transform(X_test[[col]])

    cat_cols =  set(X_train.columns) - set(num_cols)
    for col in cat_cols:
        mode_val = X_train[col].value_counts().index[0]
        X_train[col] = X_train[col].fillna(mode_val)
        mode_val = X_test[col].value_counts().index[0]
        X_test[col] = X_test[col].fillna(mode_val)
    return X,y,X_train,X_test,y_train,y_test


