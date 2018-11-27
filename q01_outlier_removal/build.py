# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    q1=loan_data['ApplicantIncome'].quantile(0.95)
    q2=loan_data['CoapplicantIncome'].quantile(0.95)
    q3=loan_data['LoanAmount'].quantile(0.95)
    df =loan_data.drop(loan_data[(loan_data['ApplicantIncome']>q1)].index)
    df1=df.drop(df[(df['CoapplicantIncome']>q2)].index)
    df2=df1.drop(df1[(df1['LoanAmount']>q3)].index)
    return df2


