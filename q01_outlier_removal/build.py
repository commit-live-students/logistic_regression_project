import pandas as pd
import matplotlib.pyplot as plt

loan_data = pd.read_csv('data/loan_prediction_uncleaned.csv')
loan_data = loan_data.drop('Loan_ID', 1)

def outlier_removal(data):
    series1 = data.quantile(0.95)
    for i in range(0,len(series1)-2):
        data = data.drop(data[(data[series1.index[i]] > series1[i])].index)
    return data



