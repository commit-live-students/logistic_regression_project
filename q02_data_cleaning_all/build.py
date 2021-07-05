# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(df):
    
    imp_mean = Imputer(missing_values = float('NaN'), strategy='mean')
    df['LoanAmount'] = imp_mean.fit_transform(df[['LoanAmount']]).ravel()
         
    df['Gender']  = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
        
    X=df.drop(['LoanAmount'],axis=1)
    y=df['LoanAmount']
  
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.25, random_state = 9)
    
    return (X,y,X_train, X_test, y_train, y_test)



