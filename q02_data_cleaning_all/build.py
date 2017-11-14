import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(df):
    imp_mean = Imputer(missing_values='NaN',strategy = 'mean')
    imp_mean.fit(df[['LoanAmount']])
    df[['LoanAmount']] = imp_mean.transform(df[['LoanAmount']])

    cols = ['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']

    for i in cols:
        df[i] = df[i].fillna(df[i].mode()[0])

    X,y = df.iloc[:,:-1],df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state=9)

    return X,y,X_train,X_test,y_train,y_test


# Write your solution here :
