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
loan_data.columns
def data_cleaning (housing_data):
    missinglist = list(housing_data.columns[housing_data.isnull().any()])
    cate=[]
    num=[]
    for k in missinglist :
        if (str(housing_data[k].dtypes) == 'object' ):
            housing_data[k].fillna(housing_data[k].mode()[0], inplace =True)
            cate.append(k)
        else:
            housing_data[k].fillna(housing_data[k].mean(), inplace =True)
            num.append(k)
    X=housing_data.drop('Loan_Status',1)
    y=housing_data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    return X ,y, X_train, X_test, y_train, y_test
data_cleaning(loan_data)



