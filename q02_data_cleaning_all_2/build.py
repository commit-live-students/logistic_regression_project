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
    col=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    allcolumn = X_train.columns.values
      
    df=pd.get_dummies(X_train , columns=col )
    X_train = pd.concat([X_train,df], axis=1)
    X_train = X_train.drop(col,axis=1)
    X_train = X_train.drop(allcolumn,axis=1)
    X_train = X_train.drop('Dependents_3+',axis=1)
    
    df1=pd.get_dummies(X_test , columns=col )
    X_test = pd.concat([X_test,df1], axis=1)
    X_test = X_test.drop(col,axis=1)
    X_test = X_test.drop(allcolumn,axis=1)
    X_test = X_test.drop('Dependents_3+',axis=1)
    
    return X_train ,X_test,y_train,y_test




