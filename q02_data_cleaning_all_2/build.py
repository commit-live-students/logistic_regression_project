# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from math import sqrt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

# Write your solution here :
def data_cleaning_2(X_train,X_test,y_train,y_test):
    for x in ['CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']:
    
        X_train[x] = np.sqrt(X_train[x])  
        X_test[x]= np.sqrt(X_test[x])
    
    X_train1 = pd.get_dummies(X_train)
    X_test1 = pd.get_dummies(X_test)
    
    X_train1=X_train1.drop(['Dependents_0','Gender_Female','Education_Graduate','Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    
    X_test1=X_test1.drop(['Dependents_0','Gender_Female','Education_Graduate','Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    
    return X_train1,X_test1,y_train,y_test

data_cleaning_2(X_train, X_test, y_train, y_test)


y_train


