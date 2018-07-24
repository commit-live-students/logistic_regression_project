# %load q03_logistic_regression/build.py
# Default Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from logistic_regression_project.q01_outlier_removal.build import outlier_removal
from logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from logistic_regression_project.q02_data_cleaning_all_2.build import data_cleaning_2

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
X_train, X_test, y_train, y_test = data_cleaning_2(X_train, X_test, y_train, y_test)


# Write your solution code here:
def logistic_regression(X_train, X_test, y_train, y_test):
    stand_scale= StandardScaler()
    num_cols = ['ApplicantIncome','CoapplicantIncome', 'LoanAmount']
    for col in num_cols:
        X_train.loc[:, col] = stand_scale.fit_transform(X_train.loc[:, col])
        X_test.loc[:, col] = stand_scale.fit_transform(X_test.loc[:, col])
        y_train.loc[:, col] = stand_scale.fit_transform(y_train.loc[:, col])
        y_test.loc[:, col] = stand_scale.fit_transform(y_test.loc[:, col])
        
    lr = LogisticRegression(random_state=9)
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    return cm


