# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

random.seed(9)

# Write your solution here :
def data_cleaning(loan_data):
    columns = np.array(loan_data.columns)
    
    null_values_columns=[]
    for i in range(len(columns)):
        print('-----------------')
        print(columns[i])
        print(loan_data.loc[:,columns[i]].isnull().values.any())
        if (loan_data.loc[:,columns[i]].isnull().values.any() == True):
            null_values_columns.append(columns[i])
    
    
    for i in range(len(null_values_columns)):
        dtype = loan_data.loc[:,null_values_columns[i]].get_dtype_counts().index[0]
        if dtype == 'float64':
            mean = loan_data.loc[:,null_values_columns[i]].mean()
            loan_data.loc[:,null_values_columns[i]].fillna(mean,inplace=True)
        elif dtype=='object':
            mode = loan_data.loc[:,null_values_columns[i]].mode()[0]
            loan_data.loc[:,null_values_columns[i]].fillna(mode,inplace=True)
    
    X_train,X_test,y_train,y_test = train_test_split(loan_data.iloc[:,:-1],loan_data.iloc[:,-1],test_size=0.25)
    X , y = loan_data.iloc[:,:-1],loan_data.iloc[:,-1]
    
    return X,y,X_train,X_test,y_train,y_test










