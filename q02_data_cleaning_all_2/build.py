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


# Write your solution here :
def data_cleaning_2(X_train,X_test,y_train,y_test):
    X_train = pd.get_dummies(X_train)
    X_train = np.sqrt(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = np.sqrt(X_test)
    X_train = X_train.drop(['Gender_Female','Married_No','Dependents_0','Education_Graduate','Self_Employed_No','Property_Area_Rural'],axis=1)
    X_test = X_test.drop(['Gender_Female','Married_No','Dependents_0','Education_Graduate','Self_Employed_No','Property_Area_Rural'],axis=1)
    return X_train,X_test,y_train,y_test
data_cleaning_2(X_train,X_test,y_train,y_test)


