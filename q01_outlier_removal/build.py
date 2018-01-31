# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    quantiles = data[cols].quantile(q=0.95)
    for col in cols:
        data = data.drop(data[(data[col] > quantiles[col])].index)
    return data
