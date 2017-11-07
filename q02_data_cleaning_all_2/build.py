# %load q02_data_cleaning_all_2/build.py
# Default Imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)

# Write your solution here :
def data_cleaning_2(X_train,X_test,y_train,y_test):
    numeric_df=X_train._get_numeric_data()
#a= list(set(X_train.columns) -set(X_train._get_numeric_data().columns))
    categorical_df=X_train[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
    sca_num_df=np.sqrt(numeric_df)
    sca_cat_df=pd.get_dummies(categorical_df)
    sca_cat_df=sca_cat_df.drop('Dependents_3+',axis=1)

    numeric_test_df=X_test._get_numeric_data()
#a= list(set(X_train.columns) -set(X_train._get_numeric_data().columns))
    categorical_test_df=X_test[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
    sca_num_test_df=np.sqrt(numeric_test_df)
    sca_cat_test_df=pd.get_dummies(categorical_test_df)
    sca_cat_test_df=sca_cat_test_df.drop('Dependents_3+',axis=1)
    return sca_cat_df,sca_cat_test_df,y_train,y_test
