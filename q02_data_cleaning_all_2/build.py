# Default Imports

import pandas as pd
import numpy as np

from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

cols = X_train.columns
miss_col = X_train.columns[X_train.isnull().any()]
num_col = X_train._get_numeric_data().columns
cat_col = list(set(cols) - set(num_col))

# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    X_train[num_col]=np.sqrt(X_train[num_col])
    X_test[num_col]=np.sqrt(X_test[num_col])

    X_t_d=pd.get_dummies(X_train[cat_col], drop_first=False)
    X_te_d=pd.get_dummies(X_test[cat_col], drop_first=False)
    X_train=pd.concat([X_train,X_t_d],axis=1)
    return X_train['Dependents_1'].value_counts()
