# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import LabelEncoder
import warnings

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    num_cols = X_train._get_numeric_data().columns
    for col in num_cols:
        X_train[col] = np.sqrt(X_train[col])
        X_test[col] = np.sqrt(X_test[col])
    cat_data = X_train[['Self_Employed', 'Married', 'Dependents', 'Gender', 'Property_Area', 'Education']]
    enc_res = pd.get_dummies(cat_data,drop_first=True)
    X_train = X_train.join(enc_res)
    for col in ['Self_Employed', 'Married', 'Dependents', 'Gender', 'Property_Area', 'Education']:
        X_train.drop(col, axis=1, inplace=True)
    cat_data = X_test[['Self_Employed', 'Married', 'Dependents', 'Gender', 'Property_Area', 'Education']]
    enc_res = pd.get_dummies(cat_data,drop_first=True)
    X_test = X_test.join(enc_res)
    for col in ['Self_Employed', 'Married', 'Dependents', 'Gender', 'Property_Area', 'Education']:
        X_test.drop(col, axis=1, inplace=True)
    return X_train, X_test, y_train, y_test


