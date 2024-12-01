# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal



loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
#loan_dathttps://app2.commit.live/program/d96bd5ca-13fc-4eb5-8180-3fd7cbc0862b/detail/logistic_regression_project:q02_data_cleaning_all#data-files-panela = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(loan_data):
    X = loan_data.iloc[:,:-1]
    y=loan_data['Loan_Status']
    X['LoanAmount']=X['LoanAmount'].fillna(np.mean(X['LoanAmount']))
    X['Gender'].count()
    #n1=np.array(loan_data['Credit_History'])
    #unique_elements, counts_elements = np.unique(n1, return_counts=True)
    X['Gender']=X['Gender'].fillna('Female')
    X['Married']=X['Married'].fillna('Yes')
    X['Dependents']=X['Dependents'].fillna('3+')
    X['Self_Employed']=X['Self_Employed'].fillna('No')
    X['Loan_Amount_Term']=X['Loan_Amount_Term'].fillna('360')
    X['Credit_History']=X['Credit_History'].fillna('1')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    return(X,y,X_train, X_test, y_train, y_test)


data_cleaning(loan_data)


