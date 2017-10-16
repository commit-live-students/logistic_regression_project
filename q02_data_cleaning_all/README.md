# Data Cleaning

In the previous task we have removed the outlier and now we will do rest of the cleaning.

## Write a function `data_cleaning` that :
- Split the `Dependent Variables` and `Independent Variable` into train and test part. (Hint: X_train, y_train, remember!)
-`Impute` the missing value of Numerical and categorical variables in train and test part of the data.
 
### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| data | pandas DataFrame| compulsory |  | Data at hand for cleaning|


### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| X | DataFrame | Dataframe containing feature variables |
| y | Series/DataFrame | Target Variable |
| X_train | Numpy arrays for training any format acceptable by sklearn| scaled X_train |
| X_test | Numpy arrays for testing any format acceptable by sklearn| scaled X_test |
| y_train | Numpy arrays for training any format acceptable by sklearn   | y_train |
| y_test |  Numpy arrays for testing any format acceptable by sklearn   | y_test |

Hint : 
- Set random state as 9 while splitting the data set.
- Numerical variable (`LoanAmount`) imputation can be performed with mean imputation.
- Categorical variables null values should be imputed with mode imputation.

Let's get started !
