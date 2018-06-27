# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer
loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(data):
    np.random.seed(9)
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    imputer = Imputer(missing_values=np.nan, strategy='mean')
    imput = imputer.fit(data[['LoanAmount']])
    data[['LoanAmount']] = imput.transform(data[['LoanAmount']])
    
    data[['Gender']] = (data[['Gender']].fillna(data[['Gender']].mode(), inplace = True))
    data[['Married']] = (data[['Married']].fillna(data[['Married']].mode(), inplace = True))
    data[['Dependents']] = (data[['Dependents']].fillna(data[['Dependents']].mode(), inplace = True))
    data[['Self_Employed']] = (data[['Self_Employed']].fillna(data[['Self_Employed']].mode(), inplace = True))
    data[['Loan_Amount_Term']] = (data[['Loan_Amount_Term']].fillna(data[['Loan_Amount_Term']].mode(), inplace = True))
    data[['Credit_History']] = (data[['Credit_History']].fillna(data[['Credit_History']].mode(), inplace = True))
    
    return X, y, X_train, X_test, y_train, y_test
    
data_cleaning(loan_data)
    
# Write your solution here :


