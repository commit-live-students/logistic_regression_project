# %load q03_logistic_regression/build.py
# Default Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q02_data_cleaning_all_2.build import data_cleaning_2

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)
X_train, X_test, y_train, y_test = data_cleaning_2(X_train, X_test, y_train, y_test)


# Write your solution code here:
def logistic_regression(X_train, X_test, y_train, y_test):
    
    l = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    Sc = StandardScaler()
    X_train.loc[:,l] =  Sc.fit_transform(X_train.loc[:,l])
    X_test.loc[:,l] = Sc.fit_transform(X_test.loc[:,l])

    ls = LogisticRegression(random_state=9)
    ls.fit(X_train,y_train)
    y_pred = ls.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    return cm


#logistic_regression(X_train, X_test, y_train, y_test)


