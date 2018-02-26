# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

np.random.seed(9)
# Write your solution here :
def data_cleaning(data):
    X=data.iloc[:,:-1]
    y=data.Loan_Status
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=9)
    X_train['LoanAmount']=X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
    X_test['LoanAmount']= X_test['LoanAmount'].fillna( X_test['LoanAmount'].mean())

    columns=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for col in columns:
        X_train[col]=X_train[col].fillna(X_train[col].value_counts().index[0])
        X_test[col]=X_test[col].fillna(X_test[col].value_counts().index[0])
    return X, y, X_train, X_test, y_train, y_test
