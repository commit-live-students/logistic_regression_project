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
def data_cleaning(data):
    np.random.seed(9)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    X_train_loanamt = X_train[X_train['LoanAmount'].notnull()]['LoanAmount'].mean()
    X_train.loc[X_train['LoanAmount'].isnull(), 'LoanAmount'] = X_train_loanamt
    X_test.loc[X_test['LoanAmount'].isnull(), 'LoanAmount'] = X_train_loanamt
    cat = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for val in cat:
        temp = X_train[X_train[val].notnull()][val].mode()
        X_train.loc[X_train[val].isnull(), val] = temp
        X_test.loc[X_test[val].isnull(), val] = temp

    return X, y, X_train, X_test, y_train, y_test



