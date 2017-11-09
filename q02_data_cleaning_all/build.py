# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
np.random.seed(9)

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    X = loan_data.iloc[:, :-1]
    y = loan_data['Loan_Status']

    X_train, X_test = train_test_split(X, test_size=0.25, random_state=9)
    y_train, y_test = train_test_split(y, test_size=0.25, random_state=9)

    X_train['LoanAmount'] = X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
    X_train['Gender'] = X_train['Gender'].fillna(X_train['Gender'].mode()[0])
    X_train['Married'] = X_train['Married'].fillna(X_train['Married'].mode()[0])
    X_train['Dependents'] = X_train['Dependents'].fillna(X_train['Dependents'].mode()[0])
    X_train['Self_Employed'] = X_train['Self_Employed'].fillna(X_train['Self_Employed'].mode()[0])
    X_train['Loan_Amount_Term'] = X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mode()[0])
    X_train['Credit_History'] = X_train['Credit_History'].fillna(X_train['Credit_History'].mode()[0])

    X_test['LoanAmount'] = X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())
    X_test['Gender'] = X_test['Gender'].fillna(X_test['Gender'].mode()[0])
    X_test['Married'] = X_test['Married'].fillna(X_test['Married'].mode()[0])
    X_test['Dependents'] = X_test['Dependents'].fillna(X_test['Dependents'].mode()[0])
    X_test['Self_Employed'] = X_test['Self_Employed'].fillna(X_test['Self_Employed'].mode()[0])
    X_test['Loan_Amount_Term'] = X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mode()[0])
    X_test['Credit_History'] = X_test['Credit_History'].fillna(X_test['Credit_History'].mode()[0])

    return X,y,X_train,X_test,y_train,y_test
