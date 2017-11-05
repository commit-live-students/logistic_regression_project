import pandas as pd


loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

# Write your Solution here:
def outlier_removal(loan_data):
    quan_ai = loan_data['ApplicantIncome'].quantile(.95)
    quan_ci = loan_data['CoapplicantIncome'].quantile(.95)
    quan_la = loan_data['LoanAmount'].quantile(.95)

    #loan_data = loan_data.drop(loan_data[(loan_data['ApplicantIncome']>loan_data['ApplicantIncome'].quantile(.95)) | (loan_data['CoapplicantIncome']>loan_data['CoapplicantIncome'].quantile(.95)) | (loan_data['LoanAmount']>loan_data['LoanAmount'].quantile(.95))].index)
    loan_data = loan_data.drop(loan_data[(loan_data['ApplicantIncome']> quan_ai) |
                                        (loan_data['CoapplicantIncome']> quan_ci) |
                                        (loan_data['LoanAmount']> quan_la)].index)
    return loan_data
outlier_removal(loan_data)
