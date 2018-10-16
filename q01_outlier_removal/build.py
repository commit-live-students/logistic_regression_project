#%load q01_outlier_removal/build.py

def outlier_removal(loan_data):
    col = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    iqr = list()
    for c in col:
        iqr.append([c,loan_data[c].quantile(0.95)])
    for c in iqr:
        loan_data = loan_data.drop(loan_data[(loan_data[c[0]]>c[1])].index)
    return loan_data

