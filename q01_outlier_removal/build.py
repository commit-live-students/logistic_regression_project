# %load q01_outlier_removal/build.py
# Default imports
from greyatomlib.logistic_regression_project.q01_outlier_removal.build import outlier_removal
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(df):
    qv = 0.95
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    df_qv = df.quantile(q=qv, axis=0, numeric_only=True, interpolation='linear')
    for feature in numeric_features:
        df = df.drop(df[df[feature] > df_qv[feature]].index)
    return df




