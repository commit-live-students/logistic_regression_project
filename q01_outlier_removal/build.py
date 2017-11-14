# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(df):
    df_numeric = df[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    quantiles = df_numeric.quantile(0.95)
    for col in df.columns:
        if col in quantiles:
            df = df.drop(df[(df[col] > quantiles[col])].index)
    return df
