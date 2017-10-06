# Logistic Regression (Model Building and Fitting)

* If you have reached here so that means you have tackled the most difficult task for this data, and believe me, I appreciate your efforts.
* So, what next? Here we go...

**After you get your clean parameters, i.e. X_train, X_teest, y_train, y_test, you will build a model `Logistic Regression` on train part
and fit that model on test part.**

if this is confusing then go back to class slides and revise it once more. This is gonna be easy for you.

**CREATE A FUNCTION `logistic_regression` THAT PERFORMS THE FOLLOWING TASK:**

- Build the logistic regression model
- fit that model on the test part
- gives the confusion matrix as evaluation metric of how good fit model is. 

Below is parameters are given to include in function, and expected output result is also given.

#### Parameters:

| Parameter | dtype | argument type | default value | description |
| :---: | :---: | :---: | :---: | :---: |
| X_train | Numpy arrays for training, testing; any format acceptable by sklearn| | compulsory|scaled X_train |
| X_test | Numpy arrays for training, testing; any format acceptable by sklearn| | compulsory|scaled X_test |
| y_train | Numpy arrays for training, testing; any format acceptable by sklearn | | compulsory| y_train |
| y_test |  Numpy arrays for training, testing; any format acceptable by sklearn|  | compulsory| y_test |


#### Returns:

| Parameter | dtype  | description |
| :---: | :---: |:---: |
| cm | array of matrix | accuracy of created model |