import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
col1 = list(loan_data.columns.values)
col1.remove('Loan_Status')

def data_cleaning(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 9)
    X_train = pd.DataFrame(X_train,columns=col1)
    X_test = pd.DataFrame(X_test,columns=col1)
    X_train['LoanAmount']=X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
    X_test['LoanAmount']=X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())

    columns=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
        
    for col in columns:
        mode_1 = X_train[col].mode()
        mode_a = mode_1[0]
        X_train[col]=X_train[col].fillna(mode_a) 
        
    for col in columns:
        mode_1 = X_test[col].mode()
        mode_a = mode_1[0]
        X_test[col]=X_test[col].fillna(mode_a)         
        
    X = pd.DataFrame(X)
    return X,y,X_train,X_test,y_train,y_test


