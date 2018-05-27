# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(df):
    num_col=df[['ApplicantIncome','CoapplicantIncome',
                'LoanAmount']]
    quant=num_col.quantile(0.95)
    for col in num_col:
        df=df.drop(df[df[col]>quant[col]].index)
    return df
print(outlier_removal(loan_data))


