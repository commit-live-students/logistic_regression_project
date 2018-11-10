# %load q02_data_cleaning_all/build.py
# Default Imports
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname('__file__'))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)


# Write your solution here :
def data_cleaning(loan_data):
    X = loan_data.drop('Loan_Status',1)
    y = pd.Series(loan_data['Loan_Status'])
    X_num = X.select_dtypes(include = [np.number])
    X_num.fillna(X_num.mean(),inplace=True)
    X_cat = X.select_dtypes(exclude = [np.number])
    X_cat = X_cat.apply(lambda x:x.fillna(x.value_counts().index[0]))
    X = pd.concat([X_num,X_cat],axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=9)
    return X,y,pd.DataFrame(X_train),pd.DataFrame(X_test),y_train,y_test
data_cleaning(loan_data)


