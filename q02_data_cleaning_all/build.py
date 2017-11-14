# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(df):
    df = df.copy()
    np.random.seed(9)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    X_train = removeMissingValues(X_train)
    X_test = removeMissingValues(X_test)
    return X, y, X_train, X_test, y_train, y_test

def removeMissingValues(X):
    X = X.copy()
    df_categoric = X.select_dtypes(include=['object'])
    for col in df_categoric.columns:
        df_categoric[col] = df_categoric[col].fillna(df_categoric[col].mode()[0])

    df_numeric = X.select_dtypes(include=['float64','int64'])
    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].mode()[0])

    df_new = pd.concat([df_numeric,df_categoric],axis=1)
    return df_new
