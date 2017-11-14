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

def data_cleaning_2(X_train,X_test,y_train,y_test):

    X_train_num = X_train.select_dtypes(include=['float64'])
    X_test_num = X_test.select_dtypes(include=['float64'])

    for col in X_train_num:
        X_train_num[col] = np.sqrt(X_train_num[col])
    for col in X_test_num:
        X_test_num[col] = np.sqrt(X_test_num[col])
    cat = ['Dependents','Self_Employed','Property_Area']

    X_train_cat = pd.get_dummies(X_train[cat])
    X_test_cat = pd.get_dummies(X_test[cat])

    X_train2 = pd.concat([X_train_num,X_train_cat],axis=1)
    X_test2 = pd.concat([X_test_num,X_test_cat],axis=1)

    return X_train2,X_test2,y_train,y_test



# Write your solution here :
