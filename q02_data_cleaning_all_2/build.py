# %load q02_data_cleaning_all_2/build.py
# Default Imports
import pandas as pd
import numpy as np
from greyatomlib.logistic_regression_project.q02_data_cleaning_all.build import data_cleaning
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
from sklearn.preprocessing import LabelEncoder

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
loan_data = outlier_removal(loan_data)
X, y, X_train, X_test, y_train, y_test = data_cleaning(loan_data)


# Write your solution here :
def data_cleaning_2(X_train, X_test, y_train, y_test):
    import numpy as np
    num_df_train = X_train[['ApplicantIncome','CoapplicantIncome',
                       'LoanAmount',]]
    cat_df_train = X_train[['Gender', 'Married', 'Dependents','Education', 'Self_Employed', 
                    'Loan_Amount_Term', 'Credit_History','Property_Area']]

    num_df_test= X_test[['ApplicantIncome','CoapplicantIncome',
                       'LoanAmount']]
    cat_df_test = X_test[['Gender', 'Married', 'Dependents','Education', 'Self_Employed', 
                    'Loan_Amount_Term', 'Credit_History','Property_Area']]

    for col in num_df_train:
        num_df_train[col] = np.sqrt(num_df_train[col])
    
    for col in num_df_test:
        num_df_test[col] = np.sqrt(num_df_test[col])
    
    cat_df_train = pd.get_dummies(cat_df_train)
    cat_df_test = pd.get_dummies(cat_df_test)

    cat_df_train = cat_df_train.drop(['Dependents_0','Gender_Female','Education_Graduate',
                            'Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    cat_df_test = cat_df_test.drop(['Dependents_0','Gender_Female','Education_Graduate',
                          'Self_Employed_No','Married_No','Property_Area_Rural'],axis=1)
    
    print('num_df_train ',num_df_train.shape)
    print('cat_df_train ',cat_df_train.shape)
    print('num_df_test ',num_df_test.shape)
    print('cat_df_test ',cat_df_test.shape)
    X_train = pd.concat([num_df_train,cat_df_train],axis=1)
    X_test = pd.concat([num_df_test,cat_df_test],axis=1)
    
    print('X_train ',X_train.shape)
    print('X_test ',X_test.shape)
    return X_train, X_test, y_train, y_test








