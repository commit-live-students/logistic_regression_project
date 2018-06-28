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

def data_cleaning_2(X_train, X_test, y_train, y_test):
    numerical_variables = X._get_numeric_data().columns.values
    categorical_variables = list(set(X.columns.values) - set(X._get_numeric_data().columns.values))
  
    X_train[numerical_variables] = np.sqrt(X_train[numerical_variables])
    X_test[numerical_variables] = np.sqrt(X_test[numerical_variables])
    
    X_train1 = pd.get_dummies(X_train, columns=categorical_variables, drop_first = True)
    X_test1 = pd.get_dummies(X_test, columns=categorical_variables, drop_first = True)

    return X_train1, X_test1, y_train, y_test

data_cleaning_2(X_train, X_test, y_train, y_test)

