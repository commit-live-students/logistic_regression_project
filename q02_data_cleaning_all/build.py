# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def impute_nan(to_be_imputed):
    if isinstance(to_be_imputed,pd.core.frame.DataFrame):
        for col in to_be_imputed.columns.values:
            if to_be_imputed[col].dtypes != 'object':
                mean = to_be_imputed[col].mean()
                to_be_imputed[col].fillna(value=mean,inplace=True)
            elif to_be_imputed[col].dtypes == 'object':
                mode = to_be_imputed[col].mode()[0]
                to_be_imputed[col].fillna(value=mode,inplace=True)
    elif isinstance(to_be_imputed,pd.core.frame.Series):
        # Assuming y_train/y_test to be categorical
        mode = to_be_imputed.mode()[0]
        to_be_imputed.fillna(value=mode,inplace=True)
    return to_be_imputed

def data_cleaning(data):
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=9)

    X_train_imputed = impute_nan(X_train)
    X_test_imputed  = impute_nan(X_test)
    y_train_imputed = impute_nan(y_train)
    y_test_imputed  = impute_nan(y_test)

    return X,y,X_train_imputed,X_test_imputed,y_train_imputed,y_test_imputed
