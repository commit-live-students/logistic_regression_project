# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(data):
    quantiles_values = data.quantile(0.95)
    data = data.drop(data[data.ApplicantIncome > quantiles_values[0]].index)
    data = data.drop(data[data.CoapplicantIncome > quantiles_values[1]].index)
    data = data.drop(data[data.LoanAmount > quantiles_values[2]].index)
    return data

