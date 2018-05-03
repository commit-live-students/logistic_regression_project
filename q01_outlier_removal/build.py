# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    quantile_threshold = 0.95
    threshold_appIncome = data.ApplicantIncome.quantile(quantile_threshold)
    threshold_coAppIncome = data.CoapplicantIncome.quantile(quantile_threshold)
    threshold_LoanAmount  = data.LoanAmount.quantile(quantile_threshold)
    outlier_criteria_index = data[(data.ApplicantIncome > threshold_appIncome)\
                             | (data.CoapplicantIncome > threshold_coAppIncome)\
                             | (data.LoanAmount > threshold_LoanAmount)].index
    data.drop(labels=outlier_criteria_index, axis=0, inplace=True)
    return data
