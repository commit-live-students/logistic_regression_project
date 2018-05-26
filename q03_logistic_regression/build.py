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
    scale = StandardScaler()
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    X_train.loc[:,cols] = scale.fit_transform(X_train.loc[:,cols])
    X_test.loc[:,cols] = scale.fit_transform(X_test.loc[:,cols])
    log_reg = LogisticRegression(random_state=9)
    log_reg.fit(X_train,y_train)
    y_pred = log_reg.predict(X_test)
    return confusion_matrix(y_test,y_pred)
