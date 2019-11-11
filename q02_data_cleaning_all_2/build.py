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
    
    num_col = X_train.select_dtypes(['int','float']).columns
    X_train[num_col] = X_train[num_col].apply(np.sqrt)
    X_train['Gender'].replace({'Male':0, 'Female':1},inplace = True)
    X_train['Married'].replace({'Yes':1, 'No': 0}, inplace= True)
    X_train['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace = True)
    X_train['Self_Employed'].replace({'Yes':1, 'No':0},inplace = True)
    new_pro = pd.get_dummies(X_train['Property_Area']).reset_index()
    X_train  = X_train.reset_index()
    X_train = X_train.merge(new_pro, how='left', left_on = 'index', right_on = 'index')
    X_train.index = X_train['index']
    new_dependent = pd.get_dummies(X_train['Dependents']).reset_index()
    X_train = X_train.merge(new_dependent, how = 'left', left_on = 'index' ,right_on = 'index')
    X_train.drop(['Property_Area', 'Dependents', 'Rural','index', '0'], axis = 1, inplace = True)
    
    X_train.rename(columns={'1':'Dependents_1', '2':'Dependents_2','3+':'Dependents_3'}, inplace = True)
    X_train.rename(columns={'Semiurban':'Property_Area_Semiurban','Urban':'Property_Area_Urban'}, inplace = True)
    
    
    num_col = X_test.select_dtypes(['int','float']).columns
    X_test[num_col] = X_test[num_col].apply(np.sqrt)
    X_test['Gender'].replace({'Male':0, 'Female':1},inplace = True)
    X_test['Married'].replace({'Yes':1, 'No': 0}, inplace= True)
    X_test['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace = True)
    X_test['Self_Employed'].replace({'Yes':1, 'No':0},inplace = True)
    new_pro = pd.get_dummies(X_test['Property_Area']).reset_index()
    X_test  = X_test.reset_index()
    X_test = X_test.merge(new_pro, how='left', left_on = 'index', right_on = 'index')
    X_test.index = X_test['index']
    new_dependent = pd.get_dummies(X_test['Dependents']).reset_index()
    X_test = X_test.merge(new_dependent, how = 'left', left_on = 'index' ,right_on = 'index')
    X_test.drop(['Property_Area', 'Dependents', 'Rural','index', '0'], axis = 1, inplace = True)
    
    X_test.rename(columns={'1':'Dependents_1', '2':'Dependents_2','3+':'Dependents_3'}, inplace = True)
    X_test.rename(columns={'Semiurban':'Property_Area_Semiurban','Urban':'Property_Area_Urban'}, inplace = True)


    return X_train, X_test, y_train, y_test

# num_col = X_test.select_dtypes(['int','float']).columns
# X_test[num_col] = X_test[num_col].apply(np.sqrt)
# X_test['Gender'].replace({'Male':0, 'Female':1},inplace = True)
# X_test['Married'].replace({'Yes':1, 'No': 0}, inplace= True)
# X_test['Education'].replace({'Graduate':1, 'Not Graduate':0}, inplace = True)
# X_test['Self_Employed'].replace({'Yes':1, 'No':0},inplace = True)
# new_pro = pd.get_dummies(X_test['Property_Area']).reset_index()
# X_test  = X_test.reset_index()
# X_test = X_test.merge(new_pro, how='left', left_on = 'index', right_on = 'index')
# X_test.index = X_test['index']
# new_dependent = pd.get_dummies(X_test['Dependents']).reset_index()
# X_test = X_test.merge(new_dependent, how = 'left', left_on = 'index' ,right_on = 'index')
# X_test.drop(['Property_Area', 'Dependents', 'Rural','index', '0'], axis = 1, inplace = True)





