# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    qApplicantIncome = data['ApplicantIncome'].quantile(0.95)
    qCoapplicantIncome = data['CoapplicantIncome'].quantile(0.98)
    qLoanAmount = data['LoanAmount'].quantile(0.975)

    data = data[
        (data['ApplicantIncome'] < qApplicantIncome) & 
        (data['CoapplicantIncome'] < qCoapplicantIncome) & 
        (data['LoanAmount'] < qLoanAmount)
    ]
    return data



