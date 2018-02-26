# %load q02_data_cleaning_all/build.py
# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(loan_data):
    # Imputation Using Imputer
    imp_mean = Imputer(missing_values = 'NaN', strategy='mean')
    imp_mean.fit(loan_data[['LoanAmount']])
    loan_data['LoanAmount'] = imp_mean.transform(loan_data[['LoanAmount']])
    #print(loan_data['LoanAmount'])


    #Mode value imputation
    loan_data['Gender'] = loan_data['Gender'].fillna(loan_data['Gender'].mode()[0])
    loan_data['Married'] = loan_data['Married'].fillna(loan_data['Married'].mode()[0])
    loan_data['Dependents'] = loan_data['Dependents'].fillna(loan_data['Dependents'].mode()[0])
    loan_data['Self_Employed'] = loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0])
    loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0])
    loan_data['Credit_History'] = loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0])
    X = loan_data.iloc[:,:-1]
    y = loan_data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=0.75)
    return X, y, X_train, X_test, y_train, y_test

#data_cleaning(loan_data)

#print(loan_data['Gender'])
# Write your solution here :
