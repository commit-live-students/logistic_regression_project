# Outlier_Removal

**Before we proceed with making logistic regression you have been given the data, and your task is to clean this data**

- As the name suggest you will remove the outliers from the data.
- To do this you have to calculate quantile of the data and remove those observations which are more than qunatile values. 

**Hint: This is applicable for only numerical values(continuous).**

###### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| data | pandas DataFrame| compulsory |  | Data at hand for cleaning|

###### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| loan_data | pandas DataFrame | data without outliers |