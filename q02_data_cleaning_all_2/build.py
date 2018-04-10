# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal


loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X,y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

# Write your solution here :
def data_cleaning_2(X_train,X_test,y_train,y_test):
    
    L = X_train.select_dtypes(include=['object']).columns.values
    L1 = list(L)

    l = loan_data.select_dtypes(include=['int64','float64']).columns.values
    l1 = list(l)
    X_train[l1] = np.sqrt(X_train[l1]) 
    X_test[l1] = np.sqrt(X_test[l1])

    X_train1 = pd.get_dummies(X_train,columns=L1,drop_first = True)
    X_test1 = pd.get_dummies(X_test,columns=L1,drop_first = True)
    
    return X_train1,X_test1,y_train,y_test


#data_cleaning_2(X_train,X_test,y_train,y_test)



