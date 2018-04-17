# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(df):

    q = df.quantile(0.95)

    df = df.drop(df[df.ApplicantIncome > q[0]].index)
    df = df.drop(df[df.CoapplicantIncome > q[1]].index)
    df = df.drop(df[df.LoanAmount > q[2]].index)

    return df

print(outlier_removal(loan_data))
