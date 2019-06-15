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

def data_cleaning(df):
    
    num_cols = df._get_numeric_data().columns
    tot_cols = df.columns
    cat_cols = set(tot_cols)-set(num_cols)
    loan_data.loc[:,cat_cols]

    for col in num_cols:
        df[col].fillna(df[col].mean(),inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode(),inplace=True)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 9)
    
    return X,y,X_train,X_test,y_train,y_test






data_cleaning(loan_data)

