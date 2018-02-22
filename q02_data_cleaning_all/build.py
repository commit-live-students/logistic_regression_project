# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :

def data_cleaning(data):
    #find X and y
    X,y = data.loc[:, data.columns != 'Loan_Status'], data[ 'Loan_Status']

    #split train-test (25% test)
    X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.25, random_state=9)

    num_features =['LoanAmount']
    cat_features=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']

    #find mean of train for numeric features
    for col in num_features:
        mean_train= X_train[col].mean()
        #impute missing data in train and test  with mean
        X_train.loc[:,col]=X_train[col].fillna(mean_train)
        X_test.loc[:,col]=X_test[col].fillna(mean_train)

    for col in cat_features:
        #find mode of train data for categorical features
        mode_train= X_train[col].mode()[0]
        #impute missing data in train and test  with mode
        X_train.loc[:,col]=X_train[col].fillna(mode_train)
        X_test.loc[:,col]=X_test[col].fillna(mode_train)

    return X,y,X_train,X_test,y_train,y_test



print data_cleaning (loan_data)
