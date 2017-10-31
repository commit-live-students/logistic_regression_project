import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
def outlier_removal(loan_data):
    high = .95
    quant_df = loan_data.quantile(high)
    df= loan_data.drop(loan_data[(loan_data['ApplicantIncome']>quant_df['ApplicantIncome'])|
                                 (loan_data['CoapplicantIncome']> quant_df['CoapplicantIncome'])|
                                 (loan_data['LoanAmount']> quant_df['LoanAmount'])].index)

    return df
