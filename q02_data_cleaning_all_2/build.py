# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

def data_cleaning_2(X_train, X_test, y_train, y_test):

    cat = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    all_col = X_train.columns.values

    num = [x for x in all_col if x not in cat]
    print num
    X_train[num] = np.sqrt(X_train[num])

    X_etrain = pd.get_dummies(X_train,columns = cat,drop_first = True)
    X_etest = pd.get_dummies(X_test, columns = cat, drop_first = True)

    return X_etrain, X_etest, y_train, y_test

# Write your solution here :
