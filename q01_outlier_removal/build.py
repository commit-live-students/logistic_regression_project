# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(data):
    filter1 = data[['ApplicantIncome','CoapplicantIncome','LoanAmount']].quantile(0.95)

    loan_data = data.drop(data[(data['ApplicantIncome'] > filter1['ApplicantIncome']) |
                                (data['CoapplicantIncome'] > filter1['CoapplicantIncome']) |
                                (data['LoanAmount'] > filter1['LoanAmount'])].index)

    return loan_data
