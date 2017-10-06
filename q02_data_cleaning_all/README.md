# Data Cleaning

- In the previous task we have removed the outlier and now we will do rest of the cleaning.
- In this cleaning you will perform following task:     (Hint: This is not gonna be easy!)
    - Separate variable of the data such as you will have `Independent Variables` and `Dependent Variable`.
    - Split the `Dependent Variables` and `Independent Variables` into train and test part. (Hint: X_train, y_train, remember!)
    - `Impute` the missing value of Numerical and categorical variables in train and test part of the data.
    - Perform the `SQRT transformation` on Numerical variable of train and test part of the data.
    - and at last do the `Encoding` of Categorical variables.
 
**CREATE A SINGLE FUNCTION WHICH WILL DO ALL THE ABOVE TASK AT ONCE**

#### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| data | pandas DataFrame| compulsory |  | Data at hand for cleaning|


#### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| X | DataFrame | Dataframe containing feature variables |
| y | Series/DataFrame | Target Variable |
| X_train | Numpy arrays for training, testing; any format acceptable by sklearn| scaled X_train |
| X_test | Numpy arrays for training, testing; any format acceptable by sklearn| scaled X_test |
| y_train | Numpy arrays for training, testing; any format acceptable by sklearn   | y_train |
| y_test |  Numpy arrays for training, testing; any format acceptable by sklearn   | y_test |



Let's write a function logistic_predictor()
