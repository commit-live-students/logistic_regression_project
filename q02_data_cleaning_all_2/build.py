# %load q02_data_cleaning_all_2/build.py
# Default Imports

import pandas as pd
import numpy as np

from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :
def data_cleaning_2 (X_train, X_test, y_train, y_test):
    numerics=['int16','int32','int64','float16','float32','float64']
    df_train=X_train.select_dtypes(include=numerics)
    df_test=X_test.select_dtypes(include=numerics)

    for i in df_train:
        df_train[i]=np.sqrt(df_train[i])

    for i in df_test:
        df_test[i]=np.sqrt(df_test[i])
    df_train_cat=X_train[['Dependents','Self_Employed','Property_Area']]
    df_train_cat=pd.get_dummies(df_train_cat)
    df_test_cat=X_test[['Dependents','Self_Employed','Property_Area']]
    df_test_cat=pd.get_dummies(df_test_cat)
    rx=pd.concat([df_train,df_train_cat],axis=1)
    ry=pd.concat([df_test,df_test_cat],axis=1)

    return rx,ry,y_train,y_test

#data_cleaning_2 (X_train, X_test, y_train, y_test)
