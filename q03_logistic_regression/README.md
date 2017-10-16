# Logistic Regression (Model Building and Fitting)

If you have reached here so that means you have tackled the most difficult task for this data, and believe me, I appreciate your efforts.

So, what next? Here we go...

## Write a function`logistic_regression` that :

- Build the logistic regression model.
- Fit that model on the test part.
- Gives the confusion matrix as evaluation metric of how good fit model is. 

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
| cm | array of matrix | Confusion matrix to evaluate your model |