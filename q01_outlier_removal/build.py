# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
    iqr = []
    for i in num_cols:
        iqr.append([i,data[i].quantile(0.95)])
    for i in iqr:
        data = data.drop(data[(data[i[0]]>i[1])].index)
    return data
outlier_removal(loan_data)

