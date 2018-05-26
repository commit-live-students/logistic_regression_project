import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

def data_cleaning_2(X_train1,X_test1,y_train1,y_test1):    
    cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

    for c in cols:
        X_train1[c]=np.sqrt(X_train1[c])
        X_test1[c]=np.sqrt(X_test1[c])

    cols1 = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']

    X_train1 = pd.get_dummies(data=X_train1, columns=cols1,drop_first=True)
    X_test1 = pd.get_dummies(data=X_test1, columns=cols1,drop_first=True)
    return X_train1, X_test1, y_train1, y_test1


