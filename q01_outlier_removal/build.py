# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    df = data.copy()
    qAI = df.ApplicantIncome.quantile(0.95)
    qCI = df.CoapplicantIncome.quantile(0.95)
    qLA = df.LoanAmount.quantile(0.95)
    df.drop(df[df.ApplicantIncome > qAI].index, inplace=True)
    df.drop(df[df.CoapplicantIncome > qCI].index, inplace=True)
    df.drop(df[df.LoanAmount > qLA].index, inplace=True)
    return df



