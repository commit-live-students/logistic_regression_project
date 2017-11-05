# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

#print(loan_data.info())
# Write your solution here :

def data_cleaning(loan_data):
    X=loan_data.iloc[:,:-1]
    y=loan_data['Loan_Status']

    ## imputing numerical data
    imp_loanAmount=Imputer(missing_values= 'NaN', strategy='mean')
    imp_loanAmount.fit(X[['LoanAmount']])
    X['LoanAmount']=imp_loanAmount.transform(X[['LoanAmount']])
    #print(X[['LoanAmount']])

    cat_list=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for cat in cat_list:
        X[cat]=X[cat].fillna(X[cat].mode()[0])
    #print(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=9)
    #print(X_train.isnull().values.any())
    return X,y,X_train,X_test,y_train,y_test

# X, y, X_train, X_test, y_train, y_test=data_cleaning(loan_data)
# print(X_test.isnull().values.any())
# print(X_train.isnull().values.any())
# print(type(X_train))
# print(type(X_test))
# print(type(X))
# print(type(y))

# print(type(y_train))
