# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

print(X_train.shape)

# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    
    finalX = list()
    for data in [X_train, X_test]:
        numeric_data = data._get_numeric_data()
        categorical_data = data[list(set(data.columns) - set(numeric_data.columns))]
        
        for column in numeric_data.columns:
            numeric_data[column] = numeric_data[column].apply(np.sqrt)
            
        for column in categorical_data.columns:
            le = LabelEncoder()
            categorical_data[column] = le.fit_transform(categorical_data[column])
        
        data = pd.concat([numeric_data, categorical_data], 1)
        print(data.shape)
        finalX.append(data)
    
    X_train = finalX[0]
    X_test = finalX[1]
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train))
    y_test = pd.Series(le.fit_transform(y_test))
    
    return X_train, X_test, y_train, y_test
data_cleaning_2(X_train, X_test, y_train, y_test)


