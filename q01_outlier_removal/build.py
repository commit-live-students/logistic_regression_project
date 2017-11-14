# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(data):

    ai = data['ApplicantIncome'].quantile(0.95)
    ci = data['CoapplicantIncome'].quantile(0.95)
    la = data['LoanAmount'].quantile(0.95)

    loan_data_f = data.drop(data[(data['ApplicantIncome']>ai) | (data['CoapplicantIncome'] > ci) | (data['LoanAmount'] > la)].index)


    return loan_data_f


outlier_removal(loan_data).shape



# Write your Solution here:
