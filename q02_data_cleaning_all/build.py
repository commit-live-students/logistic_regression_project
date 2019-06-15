# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)

def data_cleaning(data):
    df = data
    imputer_mean = Imputer(missing_values='NaN', strategy='mean')
    imputer_mean.fit(df[['LoanAmount']])
    df['LoanAmount'] = imputer_mean.transform(df[['LoanAmount']])
    cat_features = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for feature in cat_features:
        df[feature] = df[feature].fillna(df[feature].mode()[0])
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    np.random.seed(9)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, train_size=0.75)
    return X, y, X_train, X_test, y_train, y_test

data_cleaning(loan_data)



