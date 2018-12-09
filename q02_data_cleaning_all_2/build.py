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


# # Write your solution here :

# def data_cleaning_2(X_train,X_test, y_train, y_test):

#     num_train_cols = X_train._get_numeric_data().columns
#     tot_train_cols = X_train.columns
#     cat_train_cols = set(tot_train_cols)-set(num_train_cols)

#     for col in num_train_cols:
#         X_train[col]=np.sqrt(X_train[col])
        
#     label_encoder = LabelEncoder()
#     X_train['Gender'] = label_encoder.fit_transform(X_train['Gender'])
#     X_train['Married'] = label_encoder.fit_transform(X_train['Married'])
#     X_train['Education'] = label_encoder.fit_transform(X_train['Education'])
#     X_train['Self_Employed'] = label_encoder.fit_transform(X_train['Self_Employed'])
    
    

#     for col in cat_train_cols:
#         X_train = pd.get_dummies(X_train, columns=[col],drop_first=False)

#     #For Test
#     num_test_cols = X_test._get_numeric_data().columns
#     tot_test_cols = X_test.columns
#     cat_test_cols = set(tot_test_cols)-set(num_test_cols)

#     for col in num_test_cols:
#         X_test[col]=np.sqrt(X_test[col])

#     for col in cat_test_cols:
#         X_test = pd.get_dummies(X_test, columns=[col],drop_first=False)
        
#     X_test['Gender'] = label_encoder.fit_transform(X_test['Gender'])
#     X_test['Married'] = label_encoder.fit_transform(X_test['Married'])
#     X_test['Education'] = label_encoder.fit_transform(X_test['Education'])
#     X_test['Self_Employed'] = label_encoder.fit_transform(X_test['Self_Employed'])

#     return X_train,X_test,y_train, y_test

    

# data_cleaning_2(X_train,X_test, y_train, y_test)
def data_cleaning_2(X_train,X_test,y_train,y_test):
    X_train['ApplicantIncome']=np.sqrt(X_train['ApplicantIncome'])
    X_test['ApplicantIncome']=np.sqrt(X_test['ApplicantIncome'])
    X_train['CoapplicantIncome']=np.sqrt(X_train['CoapplicantIncome'])
    X_test['CoapplicantIncome']=np.sqrt(X_test['CoapplicantIncome'])
    X_train['LoanAmount']=np.sqrt(X_train['LoanAmount'])
    X_test['LoanAmount']=np.sqrt(X_test['LoanAmount'])
    
    lablel_encoder = LabelEncoder()
    X_train['Gender'] = lablel_encoder.fit_transform(X_train['Gender'])
    X_train['Married'] = lablel_encoder.fit_transform(X_train['Married'])
    X_train['Education'] = lablel_encoder.fit_transform(X_train['Education'])
    X_train['Self_Employed'] = lablel_encoder.fit_transform(X_train['Self_Employed'])
    
    X_test['Gender'] = lablel_encoder.fit_transform(X_test['Gender'])
    X_test['Married'] = lablel_encoder.fit_transform(X_test['Married'])
    X_test['Education'] = lablel_encoder.fit_transform(X_test['Education'])
    X_test['Self_Employed'] = lablel_encoder.fit_transform(X_test['Self_Employed'])
    
    
    numericals_train = X_train.select_dtypes(include=[np.number])
    categoricals_train = X_train.select_dtypes(exclude=[np.number])
    dummies_train=pd.get_dummies(categoricals_train)
    dummies_train_1=dummies_train.loc[:,'Dependents_0':'Dependents_3+']
    dummies_train_2=dummies_train.loc[:,'Property_Area_Rural':'Property_Area_Urban']
    dummies_train_final=pd.concat([dummies_train_1,dummies_train_2],axis=1)
    final_X_train=pd.concat([X_train, dummies_train_final], axis = 1)
    
    final_X_train=final_X_train.drop('Dependents',axis=1)
    final_X_train=final_X_train.drop('Property_Area',axis=1)
    final_X_train=final_X_train.drop('Credit_History',axis=1)
    final_X_train=final_X_train.drop('Loan_Amount_Term',axis=1)
    
    numericals_test = X_test.select_dtypes(include=[np.number])
    categoricals_test = X_test.select_dtypes(exclude=[np.number])
    dummies_test=pd.get_dummies(categoricals_test)
    dummies_test_1=dummies_test.loc[:,'Dependents_0':'Dependents_3+']
    dummies_test_2=dummies_test.loc[:,'Property_Area_Rural':'Property_Area_Urban']
    dummies_test_final=pd.concat([dummies_test_1,dummies_test_2],axis=1)
    final_X_test=pd.concat([X_test, dummies_test_final], axis = 1)
    
    final_X_test=final_X_test.drop('Dependents',axis=1)
    final_X_test=final_X_test.drop('Property_Area',axis=1)
    final_X_test=final_X_test.drop('Credit_History',axis=1)
    final_X_test=final_X_test.drop('Loan_Amount_Term',axis=1)
    
    
    return final_X_train,final_X_test,y_train,y_test

