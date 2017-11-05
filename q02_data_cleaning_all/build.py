# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:

def data_cleaning(df2):


    imp_mean = Imputer(missing_values = 'NaN', strategy='mean')
    imp_mean.fit(df2[['LoanAmount']])
    df2[['LoanAmount']] = imp_mean.transform(df2[['LoanAmount']])

    df2['Gender'] = df2['Gender'].fillna(df2['Gender'].mode()[0])
    df2['Married'] = df2['Married'].fillna(df2['Married'].mode()[0])

    df2['Dependents'] = df2['Dependents'].fillna(df2['Dependents'].mode()[0])

    df2['Self_Employed'] = df2['Self_Employed'].fillna(df2['Self_Employed'].mode()[0])

    df2['Loan_Amount_Term'] = df2['Loan_Amount_Term'].fillna(df2['Loan_Amount_Term'].mode()[0])
    df2['Credit_History'] = df2['Credit_History'].fillna(df2['Credit_History'].mode()[0])

    X, y = df2.iloc[:,:-1], df2.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)

    return X,y, X_train, X_test, y_train, y_test
