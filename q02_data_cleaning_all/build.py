# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    
    loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean())
    
    columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']

    for i in columns:
        loan_data[i] = loan_data[i].fillna(loan_data[i].mode()[0])

    
    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=9)
    
    return X, y, X_train, X_test, y_train, y_test
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean())
columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']

for i in columns:
    loan_data[i] = loan_data[i].fillna(loan_data[i].mode()[0])

X = loan_data.iloc[:,:-1]
y = loan_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=9)
type(X)

