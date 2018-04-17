# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from q02_data_cleaning_all.build import data_cleaning
#from q01_outlier_removal.build import outlier_removal

def outlier_removal(data):
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for val in num_features:
        qtemp = data[val].quantile(0.9609)
        data.drop(labels = data.loc[data.loc[:,val] > qtemp, val].index.values, axis=0, inplace=True)

    return data

def data_cleaning(data):
    np.random.seed(9)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
    X_train_loanamt = X_train[X_train['LoanAmount'].notnull()]['LoanAmount'].mean()
    X_train.loc[X_train['LoanAmount'].isnull(), 'LoanAmount'] = X_train_loanamt
    X_test.loc[X_test['LoanAmount'].isnull(), 'LoanAmount'] = X_train_loanamt
    cat = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for val in cat:
        temp = X_train[X_train[val].notnull()][val].mode()[0]
        X_train.loc[X_train[val].isnull(), val] = temp
        X_test.loc[X_test[val].isnull(), val] = temp

    return X, y, X_train, X_test, y_train, y_test

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    num = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for val in num:
        X_train[val] = np.sqrt(X_train[val])
        X_test[val] = np.sqrt(X_test[val])
    #['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    X_train = pd.get_dummies(X_train, columns=['Dependents','Property_Area'])
    X_test = pd.get_dummies(X_test, columns=['Dependents','Property_Area'])
    X_train.drop(labels=['Dependents_0', 'Property_Area_Rural'], axis=1, inplace=True)
    X_test.drop(labels=['Dependents_0', 'Property_Area_Rural'], axis=1, inplace=True)
    #X_train.drop(labels=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], axis=1, inplace=True)
    #X_test.drop(labels=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], axis=1, inplace=True)
    return X_train, X_test, y_train, y_test
