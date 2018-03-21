# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(data):
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for val in num_features:
        qtemp = data[val].quantile(0.9609)
        data.drop(labels = data.loc[data.loc[:,val] > qtemp, val].index.values, axis=0, inplace=True)

    return data
