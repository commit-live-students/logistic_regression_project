# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(dataset):
    qual_vals = dataset.quantile(0.95)

    dataset =dataset.drop(dataset[dataset['ApplicantIncome'] > qual_vals[0]].index)
    dataset= dataset.drop(dataset[dataset['CoapplicantIncome'] > qual_vals[1]].index)
    dataset= dataset.drop(dataset[dataset['LoanAmount'] > qual_vals[2]].index)

    return dataset
