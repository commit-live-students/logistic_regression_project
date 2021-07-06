# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(df):
    q_vals = df.select_dtypes(include = ['int64', 'float64']).quantile(0.95)
    num_feats = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for i in num_feats:
        df = df.drop(df[df[i] > q_vals[i]].index)
    print(df.shape)
    return df
q_vals.index.values.tolist()
outlier_removal(loan_data)



