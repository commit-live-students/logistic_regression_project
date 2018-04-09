# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(df):
    l = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    quant = df.quantile(0.95)

    for i in l:
        df = df.drop(df[df[i] > quant[i]].index)


    return df

#print(outlier_removal(loan_data))
