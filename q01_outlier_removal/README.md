# OutlierRemoval

Before we proceed with making logistic regression you have been given the data, and your task is to clean this data.

## Write a function `outlier_removal` that :
- Calculates quantile of the data and remove those observations which are more than quantile values. 

Hint: 
- This is applicable for only numerical values(continuous).
- This data contains `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount` as numerical features.
- Keep the quantile value as 0.95 and remove those data points which are more than 0.95 quantile..

### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| data | pandas DataFrame| compulsory |  | Data at hand for cleaning|

### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| loan_data | pandas DataFrame | data without outliers |
