# Default Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import Imputer

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def imputer(dataset,miss_col,num_col,cat_col):
    for x in miss_col:
            if x in num_col:
                #print x
                fill_mean = Imputer(missing_values='NaN', strategy='mean')
                fill_mean.fit(dataset[[x]])
                dataset[[x]] = fill_mean.transform(dataset[[x]])
                #print(dataset[x])
                dataset[x].fillna(dataset.mean())
                #ans= dataset[x].isnull().values.any()
                #print ans
            else:
                dataset[x] = dataset[x].fillna(dataset[x].mode()[0])
                cat_df = dataset[x]
                #print(dataset[x])
                #ans= dataset[x].isnull().values.any()
                #print ans

    return dataset

def data_cleaning(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
    cols=X_train.columns
    miss_cols=X_train.columns[X_train.isnull().any()]
    num_col = X_train._get_numeric_data().columns
    cat_col = list(set(cols) - set(num_col))
    X_train=imputer(X_train,miss_cols,num_col,cat_col)

    cols=X_test.columns
    miss_cols=X_test.columns[X_test.isnull().any()]
    num_col = X_test._get_numeric_data().columns
    cat_col = list(set(cols) - set(num_col))
    X_test=imputer(X_test,miss_cols,num_col,cat_col)

    return X,y,X_train,X_test,y_train,y_test
