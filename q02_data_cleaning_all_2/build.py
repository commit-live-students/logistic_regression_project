# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

def data_cleaning_2(X_train,X_test,y_train,y_test):
    X_train["ApplicantIncome"] = np.sqrt(X_train['ApplicantIncome'])
    X_test["ApplicantIncome"]=np.sqrt(X_test["ApplicantIncome"])
    X_train= pd.get_dummies(X_train, columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"])
    X_test= pd.get_dummies(X_test, columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"])
    X_train=X_train.drop(['Dependents_0','Gender_Female','Education_Graduate','Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    X_test=X_test.drop(['Dependents_0','Gender_Female','Education_Graduate','Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    return X_train,X_test,y_train,y_test
