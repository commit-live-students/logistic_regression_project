# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    np.random.seed(9)
    X = loan_data.iloc[:,:-1]
    y = loan_data.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

    num_columns = X_train._get_numeric_data().columns
    cat_columns = set(X_train.columns) - set(X_train._get_numeric_data().columns)

    for col in num_columns:
        mean = X_train[col].mean()
        X_train[col].fillna(mean, inplace = True)
        X_test[col].fillna(mean, inplace = True)

    for col in cat_columns:
        mode = X_train[col].value_counts().index[0]
        X_train[col].fillna(mode, inplace = True)
        X_test[col].fillna(mode, inplace = True)
    
    return X,y,X_train,X_test,y_train,y_test

