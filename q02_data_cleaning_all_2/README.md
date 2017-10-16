# Data Cleaning -2

In the previous task we have splitted our dataframe into train and test and also did the missing value imputation in both numerical amd categorical variable ,now we will do rest of the cleaning.

## Write a function `data_cleaning_2` that :
- Perform the `SQRT transformation` on Numerical variable of train and test part of the data.
- and at last do the `Encoding` of Categorical variables.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| X_train | Numpy arrays for training any format acceptable by sklearn| scaled X_train |
| X_test | Numpy arrays for testing any format acceptable by sklearn| scaled X_test |
| y_train | Numpy arrays for training any format acceptable by sklearn   | y_train |
| y_test |  Numpy arrays for testing any format acceptable by sklearn   | y_test |


### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| X_train | Numpy arrays for training any format acceptable by sklearn| scaled X_train |
| X_test | Numpy arrays for testing any format acceptable by sklearn| scaled X_test |
| y_train | Numpy arrays for training any format acceptable by sklearn   | y_train |
| y_test |  Numpy arrays for testing any format acceptable by sklearn   | y_test |



Let's get started !
