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
    np.random.seed(9)
    X,y = loan_data.iloc[:,:-1],loan_data.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=9)
    loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mean())
    columns = loan_data[['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']]
    for column in loan_data.columns:
        loan_data[column].fillna(loan_data[column].mode()[0], inplace=True)
    return X,y,X_train,X_test,y_train,y_test

