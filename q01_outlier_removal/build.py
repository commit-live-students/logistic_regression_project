# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:

def outlier_removal(df):
    vQuantile = df.quantile(q=0.95)
    df = df.drop(df[(df['ApplicantIncome']>vQuantile['ApplicantIncome'])
                    | (df['CoapplicantIncome']>vQuantile['CoapplicantIncome'])
                    | (df['LoanAmount']>vQuantile['LoanAmount'])
                    ].index)
    return df

# a = outlier_removal(loan_data)
# print type(a)
# print a.shape
# print a.head()
