# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:

def outlier_removal(loan_data):
    loan_data =loan_data.drop(loan_data[(loan_data['ApplicantIncome']>14583)
         | (loan_data['CoapplicantIncome']>4997.4)
         | (loan_data['LoanAmount']>297.8)].index)

    return loan_data
