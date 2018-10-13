# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

# Write your Solution here:

#num_feature_data = loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]

# Write your code here:
def outlier_removal(loan_data):
    #df = num_feature_data
    #df = df.drop(df[(df['ApplicantIncome']>df['ApplicantIncome'].quantile(0.95)) | (df['CoapplicantIncome']>df['CoapplicantIncome'].quantile(0.95)) | (df['LoanAmount']>df['LoanAmount'].quantile(0.95))].index)

    loan_data = loan_data.drop(loan_data[(loan_data['ApplicantIncome']>loan_data['ApplicantIncome'].quantile(0.95)) | (loan_data['CoapplicantIncome']>loan_data['CoapplicantIncome'].quantile(0.95)) | (loan_data['LoanAmount']>loan_data['LoanAmount'].quantile(0.95))].index)

    #loan_data.head()

    #print (df)
    
    return loan_data
        
    
outlier_removal(loan_data)

loan_data.head()


