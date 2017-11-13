# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    df=loan_data
    df2=df.copy()
    df2["Credit_History"] = df2["Credit_History"].fillna(0)
    df2["Gender"] = df2["Gender"].fillna("Other")
    df2["Married"] = df2["Married"].fillna("Other")
    df2["Self_Employed"] = df2["Self_Employed"].fillna("No")
    df2["Dependents"] = df2["Dependents"].fillna(0)
    imp_mean.fit(df2[['LoanAmount']])
    df2['LoanAmount'] = imp_mean.transform(df2[['LoanAmount']])
    imp_mean.fit(df2[['Loan_Amount_Term']])
    df2['Loan_Amount_Term'] = imp_mean.transform(df2[['Loan_Amount_Term']])
    X = df2.iloc[:,:-1]
    y = df2.Loan_Status
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    return X, y, X_train, X_test, y_train, y_test
