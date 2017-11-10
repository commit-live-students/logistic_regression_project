# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
numerical_cols = []
for col in loan_data.columns.values:
    if loan_data[col].dtypes != 'object' and col != 'Loan_Amount_Term' and col != 'Credit_History':
        numerical_cols.append(col)
# print numerical_cols
# print loan_data.shape
# for col in numerical_cols:
#     loan_data.drop(loan_data[loan_data[col] > loan_data[col].quantile(q=0.95)].index, inplace=True)
# print loan_data.shape

def outlier_removal(loan_data):
    for col in numerical_cols:
        loan_data.drop(loan_data[loan_data[col] > loan_data[col].quantile(q=0.95)].index, inplace=True)
    return loan_data
