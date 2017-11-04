# %load q02_data_cleaning_all_2/build.py
# Default Imports

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
    #y_train=np.sqrt(y_train)
    print(X_train.shape)
    numeric_feature = [a for a in range(len(X_train.dtypes)) if X_train.dtypes[a] in ['int64','float64']]
    #print(X_train.shape)

    #numeric data transformation for test and train data
    numeric_data_train=X_train.iloc[:,numeric_feature]
    numeric_data_train_sqrt=numeric_data_train**0.5

    numeric_data_test=X_test.iloc[:,numeric_feature]
    numeric_data_test_sqrt=numeric_data_test**0.5



    #Categorical data encoding
    cat_name_train = list(X_train.select_dtypes(include=['category','object']))
    #cat_name = df.columns.difference(df.columns[numeric_feature])
    cat_data_train = X_train.loc[:,cat_name_train]
    cat_data_train_label=pd.get_dummies(cat_data_train, drop_first=True)

    cat_data_test=X_test.loc[:,cat_name_train]
    cat_data_test_label=pd.get_dummies(cat_data_test, drop_first=True)






    # final return statement
    X_train_engg=pd.concat([cat_data_train_label,numeric_data_train_sqrt],axis=1)
    X_test_engg=pd.concat([cat_data_test_label,numeric_data_test_sqrt],axis=1)

    #print(X_train_engg.shape)
    #print(X_test_engg.shape)

    return X_train_engg,X_test_engg,y_train,y_test

    #print(numeric_data_test_sqrt.shape)
    #train_val = numeric_data_train_sqrt['Dependents_1'].value_counts()
    #pd.concat(numeric_data_train_sqrt,cat_data_train,axis=0)

    #print(numeric_data_train_sqrt.shape)
    #print(numeric_data_test_sqrt.shape)

# X_train, X_test, y_train, y_test=data_cleaning_2(X_train, X_test, y_train, y_test)
# train_val= (X_train['Dependents_1'].value_counts())
# train_val1=(X_train['Property_Area_Urban'].value_counts())
# test_val=(X_test['Property_Area_Urban'].value_counts())
# test_val1=(X_test['Dependents_1'].value_counts())
# print(train_val)
# print(train_val1)
# print(test_val)
# print(test_val1)
# print(X_train.shape)
# print(X_test.shape)
