# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import LabelEncoder

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


def data_cleaning_2(X_train, X_test, y_train, y_test):
    numeric_cols = list(X_train.dtypes[X_train.dtypes != np.object].index)
    categoric_cols = list(X_train.dtypes[X_train.dtypes == np.object].index)
    X_train[numeric_cols] = X_train[numeric_cols].apply(np.sqrt)
    X_test[numeric_cols] = X_test[numeric_cols].apply(np.sqrt)
    le = LabelEncoder()
    X_train = pd.concat([X_train, pd.get_dummies(X_train[categoric_cols], drop_first=True)], axis=1)
    X_test = pd.concat([X_test, pd.get_dummies(X_test[categoric_cols], drop_first=True)], axis=1)
    X_train.drop(categoric_cols, inplace=True, axis=1)
    X_test.drop(categoric_cols, inplace=True, axis=1)
    return X_train, X_test, y_train, y_test


