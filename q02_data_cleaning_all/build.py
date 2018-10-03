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
    for x in loan_data.columns:
        if loan_data[x].dtypes == np.number:
            loan_data[x] = loan_data[x].fillna(loan_data[x].mean())
        if loan_data[x].dtypes == 'object':
            loan_data[x] = loan_data[x].fillna(loan_data[x].mode()[0])
    X = loan_data.drop('Loan_Status',axis = 1)
    y = loan_data['Loan_Status']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
    return X,y,X_train,X_test,y_train,y_test
data_cleaning(loan_data)



