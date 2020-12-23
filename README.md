House Prices Prediction
----------------------

In this project, I predicted house prices in Ames, United States, given data on 2919 houses in the city. This data comprises of 79 different variables, from the overall quality to the type of roof material. With 1460 of these properties' sale price known, the goal was to predict the other 1459 prices. 

Within the notebook, I show my steps in performing data science techniques including:

* Exploratory data analysis - understanding the data and the problem, visualising our target variable and multivariate analysis, removing outliers and imputing missing values...
* Feature engineering - understanding feature importance, performing feature selection, transforming feature types (e.g. numerical, ordinal vs nominal data)...
* Regression modelling - evaluation of performance through cross-validation, optimisation through regularisation models & grid-searching, benchmarking machine learning models...

I fit four different machine learning models (Ridge, Lasso, Elastic Net and XGBoost) and evaluated their performance by calculating their RMSE (root-mean-squared-error) between the cross-validation predictions and actual values of sale price. The final predictions for the 'unknown' sale prices yielded a RMSE of 0.13356. Towards the end of the notebook, I mention possible improvements to the model.

Details of the competition can be found at `https://www.kaggle.com/c/house-prices-advanced-regression-techniques`.

Installation
----------------------

### Download the data

* Download the ZIP file. 
* Extract `regression_house_prices-master.zip` into your desired directory.

### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt` in a command console.
	* Make sure you use Python 3.
	* You may want to use a virtual environment e.g. Anaconda Navigator.

* You will also need to have the software installed to run and execute a Jupyter Notebook. If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.

Usage
-----------------------

* Run `Regression Modelling - House Prices.ipynb` within Jupyter Notebook.
	* Make sure to run the whole notebook and restart the kernel if changes were made.
	* The notebook may take up to a few minutes to run.
