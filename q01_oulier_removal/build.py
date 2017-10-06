# Outlier removal
import pandas as pd
loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

#Write your Solution here
# Outlier Removal
