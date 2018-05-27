# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(df):
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=9)
    X_train['LoanAmount']=X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
    X_test['LoanAmount']=X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())
    
    category_col=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']
    for col in category_col:
        X_train[col]=X_train[col].fillna(X_train[col].mode())
        X_test[col]=X_test[col].fillna(X_test[col].mode())
    
    return X,y,X_train,X_test,y_train,y_test
    
print(data_cleaning(loan_data))        


