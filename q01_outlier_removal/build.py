# %load q01_outlier_removal/build.py
# Default imports
import pandas as pd

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)
quant_list=[]


# Write your Solution here:
def outlier_removal(loan_data):
    data_cols=['ApplicantIncome','CoapplicantIncome','LoanAmount']
    req_df=loan_data[['ApplicantIncome','CoapplicantIncome','LoanAmount']]
    for i in data_cols:
        quant_list.append([i,loan_data[i].quantile(0.95)])
    for j in quant_list:
        loan_data=loan_data.drop(loan_data[(loan_data[j[0]]>j[1])].index)
    return loan_data
    
     
        
    
        

    
outlier_removal(loan_data)    


