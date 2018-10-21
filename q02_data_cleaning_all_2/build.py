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
def data_cleaning_2(X_train, X_test, y_train, y_test):
    cat_col = (X_train.select_dtypes(include=['object']).columns)
    num_col = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    
    X_train['ApplicantIncome_sqrt'] = np.sqrt(X_train['ApplicantIncome'] )
    X_test['ApplicantIncome_sqrt'] = np.sqrt(X_test['ApplicantIncome'] )
    X_train['CoapplicantIncome_sqrt'] = np.sqrt(X_train['CoapplicantIncome'] )
    X_test['CoapplicantIncome_sqrt'] = np.sqrt(X_test['CoapplicantIncome'] )
    X_train['LoanAmount_sqrt'] = np.sqrt(X_train['LoanAmount'] )
    X_test['LoanAmount_sqrt'] = np.sqrt(X_test['LoanAmount'] )
    
    df_cat_train = pd.get_dummies(X_train[cat_col],drop_first=True)
    df_cat_test = pd.get_dummies(X_test[cat_col],drop_first=True)

    X_train = pd.concat([X_train,df_cat_train],axis =1)
    X_test = pd.concat([X_test,df_cat_test],axis =1)
    
    drop_col = list(cat_col) + num_col
    X_train.drop(labels=drop_col,axis=1,inplace=True)
    X_test.drop(labels=drop_col,axis=1,inplace=True)
    
    return X_train, X_test, y_train, y_test













