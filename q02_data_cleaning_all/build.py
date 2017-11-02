# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

np.random.seed(9)

def data_cleaning(loan_data):
    loan_data.sort_index(inplace=True)
    X = loan_data.iloc[:, :-1]
    y = loan_data.iloc[:, -1]

   # dfmi.loc[:,('one','second')]

    X.LoanAmount = X.LoanAmount.fillna(X.LoanAmount.mean())

    X['Gender'] = X['Gender'].fillna(X['Gender'].mode()[0])
    X['Married'] = X['Married'].fillna(X['Married'].mode()[0])
    X['Dependents'] = X['Dependents'].fillna(X['Dependents'].mode()[0])
    X['Self_Employed'] = X['Self_Employed'].fillna(X['Self_Employed'].mode()[0])
    X['Loan_Amount_Term'] = X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mode()[0])
    X['Credit_History'] = X['Credit_History'].fillna(X['Credit_History'].mode()[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

    #Return_val = X_train.isnull().values.any()
    #Return_val1 = X_test.isnull().values.any()

    #print(X)
    return  X, y, X_train, X_test, y_train, y_test


# Write your solution here :
