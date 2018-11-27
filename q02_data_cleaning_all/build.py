# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    
    categoricals = loan_data.select_dtypes(exclude=[np.number])
    numericals = loan_data.select_dtypes(include=[np.number])
    numericals['LoanAmount'].fillna(numericals['LoanAmount'].mean(),inplace=True)
    numericals['Loan_Amount_Term'].fillna(statistics.mode(numericals['Loan_Amount_Term'].values), inplace = True)
    numericals['Credit_History'].fillna(statistics.mode(numericals['Credit_History'].values), inplace = True)
    categoricals['Gender'].fillna(statistics.mode(categoricals['Gender'].values), inplace = True)
    categoricals['Married'].fillna(statistics.mode(categoricals['Married'].values), inplace = True)
    categoricals['Dependents'].fillna(statistics.mode(categoricals['Dependents'].values), inplace = True)
    categoricals['Self_Employed'].fillna(statistics.mode(categoricals['Self_Employed'].values), inplace = True)
    X=loan_data.iloc[:,:-1]
    y=loan_data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, test_size=0.25)
    return X,y,X_train,X_test,y_train,y_test
    
    


