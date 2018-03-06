# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(df):
    df = df.copy()
    num_cols = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
    quantile_95 = num_cols.quantile(0.95)
    for colu in num_cols:
        quantile = quantile_95[colu]
        #print quantile
        df=df.drop(df[df[colu]>quantile].index)
    return df
    #return num_cols.columns

print outlier_removal(loan_data)
