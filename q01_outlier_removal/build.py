# %load q01_outlier_removal/build.py
# Default imports
import numpy as np
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


def outlier_removal(data):
    quantile_value = 0.965
    applicant_income_limit = data['ApplicantIncome'].quantile(quantile_value)
    co_applicant_income_limit = data['CoapplicantIncome'].quantile(quantile_value)
    loan_amount_limit = data['LoanAmount'].quantile(quantile_value)
    return data[
        (data['ApplicantIncome'] < applicant_income_limit) &
        (data['CoapplicantIncome'] < co_applicant_income_limit) &
        (data['LoanAmount'] < loan_amount_limit)]



