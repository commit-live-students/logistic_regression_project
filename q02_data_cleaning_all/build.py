# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 9, test_size = 0.25)
    X_train_mean = X_train['LoanAmount'].mean()
    X_test_mean = X_test['LoanAmount'].mean()
    X_test.LoanAmount.fillna(X_test_mean,inplace = True)
    X_train.LoanAmount.fillna(X_train_mean,inplace = True)
    
    X_test.Gender.fillna(X_test['Gender'].mode()[0],inplace = True)
    X_test.Married.fillna(X_test['Married'].mode()[0],inplace = True)
    X_test.Dependents.fillna(X_test['Dependents'].mode()[0],inplace = True)
    X_test.Self_Employed.fillna(X_test['Self_Employed'].mode()[0],inplace = True)
    X_test.Loan_Amount_Term.fillna(X_test['Loan_Amount_Term'].mode()[0],inplace = True)
    X_test.Credit_History.fillna(X_test['Credit_History'].mode()[0],inplace = True)
    
    X_train.Gender.fillna(X_test['Gender'].mode()[0],inplace = True)
    X_train.Married.fillna(X_test['Married'].mode()[0],inplace = True)
    X_train.Dependents.fillna(X_test['Dependents'].mode()[0],inplace = True)
    X_train.Self_Employed.fillna(X_test['Self_Employed'].mode()[0],inplace = True)
    X_train.Loan_Amount_Term.fillna(X_test['Loan_Amount_Term'].mode()[0],inplace = True)
    X_train.Credit_History.fillna(X_test['Credit_History'].mode()[0],inplace = True)
    
    return (X,y,X_train,X_test,y_train,y_test)
    
  


