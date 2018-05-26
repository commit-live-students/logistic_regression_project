# Default Imports
import pandas as pd
import numpy as np
from logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    numeric_columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    cat_columns = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    # numeric columns
    for val in numeric_columns:
        X_train[val] = X_train[val].apply(np.sqrt)
        X_test[val] = X_test[val].apply(np.sqrt)
    # Categorical columns, using drop_first to remove one additional column
    # keeping k-1 column and kth column becomes redundant in case of categorical variables
    X_train = pd.get_dummies(X_train, columns=cat_columns, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_columns, drop_first=True)
    return X_train, X_test, y_train, y_test
