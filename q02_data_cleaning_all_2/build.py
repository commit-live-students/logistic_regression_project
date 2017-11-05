# Default Imports

import pandas as pd
import numpy as np

from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


def data_cleaning_2(X_train,X_test,y_train,y_test) :

    for i in X_train.select_dtypes(include=[np.number]).columns.tolist() :
        np.sqrt(X_train[i])

    for i in X_test.select_dtypes(include=[np.number]).columns.tolist() :
        np.sqrt(X_test[i])

    X_test = pd.get_dummies(X_test, drop_first = True )
    #print(X_test.shape)

    X_train = pd.get_dummies(X_train, drop_first = True )

    #X_train.drop(mylist, axis=1, inplace = True)
    #print(X_train.shape)
    #print(list(X_test['Dependents_1'].value_counts()))

    return X_train, X_test, y_train, y_test

# Write your solution here :
