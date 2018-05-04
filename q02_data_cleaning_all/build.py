# Default Imports
import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(data):
    random_seed = 9
    test_size = 0.25
    mean_loan_amt = data['LoanAmount'].mean()
    data['LoanAmount'].fillna(value=mean_loan_amt, inplace=True)
    category_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for column in category_columns:
        col_mode = data[column].mode()
        data[column].fillna(value=col_mode[0], inplace=True)
    X = data.drop(labels=['Loan_Status'],axis=1)
    y = data.Loan_Status
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    return X, y, X_train, X_test, y_train, y_test
