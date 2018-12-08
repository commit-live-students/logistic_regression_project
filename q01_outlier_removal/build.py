# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)


# Write your Solution here:
def outlier_removal(loan_data):
    
    ApplicantIncome_95   = loan_data['ApplicantIncome'].quantile(0.95)
    CoapplicantIncome_95 = loan_data['CoapplicantIncome'].quantile(0.95)
    LoanAmount_95        = loan_data['LoanAmount'].quantile(0.95)

    loan_data.drop(loan_data[loan_data['ApplicantIncome']>ApplicantIncome_95].index,inplace=True)
    loan_data.drop(loan_data[loan_data['CoapplicantIncome']>CoapplicantIncome_95].index,inplace=True)
    loan_data.drop(loan_data[loan_data['LoanAmount']>LoanAmount_95].index,inplace=True)

    return  loan_data


# loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
# loan_data = loan_data.drop('Loan_ID', 1)

#Call tothe function -
outlier_removal(loan_data)
# def outlier_removal(data):
# q1=loan_data['ApplicantIncome'].quantile(0.95)
# q2=loan_data['CoapplicantIncome'].quantile(0.95)
# q3=loan_data['LoanAmount'].quantile(0.95)

# print(q1,q2,q3)
# df =loan_data.drop(loan_data[(loan_data['ApplicantIncome']>q1)].index)
# df1=df.drop(df[(df['CoapplicantIncome']>q2)].index)
# df2=df1.drop(df1[(df1['LoanAmount']>q3)].index)
#     return df2


