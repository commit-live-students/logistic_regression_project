# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
#loan_data = outlier_removal(loan_data)
def data_cleaning(loan_data):
    X,y=loan_data.iloc[:,:-1],loan_data.iloc[:,-1]
    df=X.fillna(X.mean(),inplace=True)
    df=df[['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']].fillna(df[['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']].mode().iloc[0],inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=0.25, random_state=9)
    return df,y,X_train,X_test,y_train,y_test
