# %load q03_logistic_regression/build.py
# Default Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q02_data_cleaning_all_2.build import data_cleaning_2

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
X_train, X_test, y_train, y_test = data_cleaning_2(X_train, X_test, y_train, y_test)

def logistic_regression(X_train, X_test, y_train, y_test):
    column_transform = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
  
    stand_scale = StandardScaler()
    X_train.loc[:, column_transform] = stand_scale.fit_transform(X_train.loc[:, column_transform])
    X_test.loc[:, column_transform] = stand_scale.fit_transform(X_test.loc[:, column_transform])
        
    lr = LogisticRegression(random_state=9)
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    
    return cm


    




