# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
#loan_data.info()


#Write your solution here :
def data_cleaning(loan_data):
    
    l = ['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']
    for i in l:
        loan_data[i] = loan_data[i].fillna(loan_data[i].mode()[0])


    loan_data['LoanAmount']  = loan_data['LoanAmount'].fillna(int(loan_data['LoanAmount'].mean()))

    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
    #print(loan_data.isnull().any())
    #print(loan_data.shape)
    return X,y,X_train,X_test,y_train,y_test


#print(data_cleaning(loan_data))


