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
    numeric_feature = [a for a in range(len(X_train.dtypes)) if X_train.dtypes[a] in ['int64','float64']]
    cat_name = X_train.columns.difference(X_train.columns[numeric_feature])

    numeric_data_train_trans = np.sqrt(X_train.iloc[:,numeric_feature])
    numeric_data_test_trans = np.sqrt(X_test.iloc[:,numeric_feature])

    cat_data_train = X_train.loc[:,cat_name]
    cat_data_test = X_test.loc[:,cat_name]

    X_train_cat_encoded = pd.get_dummies(cat_data_train, drop_first=True)
    X_test_cat_encoded =  pd.get_dummies(cat_data_test, drop_first=True)


    X_train_cleaned = pd.concat([X_train_cat_encoded, numeric_data_train_trans], axis=1)
    X_test_cleaned = pd.concat([X_test_cat_encoded, numeric_data_test_trans], axis=1)

    return X_train_cleaned, X_test_cleaned, y_train, y_test
    
