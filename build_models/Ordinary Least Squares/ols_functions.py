import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from statsmodels.stats import diagnostic as diag


def exhaustive_search(X_train, y_train, pipeline_object, param_grid, cv=5, scoring='neg_mean_squared_error'):
    """
    :param X_train: feature matrix
    :param y_train: response Vector
    :param pipeline_object: the pipeline to eventually pass into GridSearchCV
    :param param_grid: all the parameters you would like to test out. All combos will be tried
    :param cv: The value of K for k-fold for cross validation when trying out each parameter combination
    :param scoring: using mean squared error (sklearn requires 'neg_mean_squared_error')
    :return: the optimal parameters found, as well as the model itself with all these parameters configured
    """
    # Run grid search on our entire preprocessing and model building pipeline
    grid = GridSearchCV(pipeline_object, param_grid, cv=cv, scoring=scoring).fit(X_train, y_train);

    # Get best parameter combo and the best model (that has all these parameters)
    return grid.best_estimator_, grid.best_params_


# ======================================================================================================================
# 1. OLS HELPER FUNCTIONS
# ======================================================================================================================
def apply_preprocessing_steps(X_train, X_test):
    X_train_sm = X_train.copy()
    X_test_sm = X_test.copy()

    # Dummy encode the status feature
    X_train_sm['Status'] = X_train_sm['Status'].apply(lambda x: 0 if x == 'Developed' else 1)
    X_test_sm['Status'] = X_test_sm['Status'].apply(lambda x: 0 if x == 'Developed' else 1)

    # Log the GDP variable and delete the original GDP variable from our dataframe copies
    X_train_sm['ln(GDP_cap)'] = np.log(X_train_sm.GDP_cap)
    X_test_sm['ln(GDP_cap)'] = np.log(X_test_sm.GDP_cap)

    X_train_sm.drop(labels='GDP_cap', axis='columns', inplace=True)
    X_test_sm.drop(labels='GDP_cap', axis='columns', inplace=True)

    # Apply knn imputation with n_neighbors = 3 and weights = uniform
    imp = KNNImputer(n_neighbors=3, weights='uniform')
    X_train_sm = pd.DataFrame(imp.fit_transform(X_train_sm), columns=X_train_sm.columns)
    X_test_sm = pd.DataFrame(imp.transform(X_test_sm), columns=X_test_sm.columns)

    # Add the column of ones for the design matrix for our training data
    design_matrix = sm.add_constant(X_train_sm)
    return design_matrix, X_test_sm


# Diagnostics:
def test_heteroskedasticity(fitted_model, alpha=0.05):
    """
    :param fitted_model: An already fitted model ready to plot out and run tests on
    :return: A scatterplot and prints the results to the Breusch Pagan test
    """
    print("Performing Breusch Pagan Test...\n")

    lagrange_multiplier, pval, _, _ = diag.het_breuschpagan(resid=fitted_model.resid, exog_het=fitted_model.model.exog)
    print(f"Lagrange Multiplier = {lagrange_multiplier}, p-value = {pval}\n")

    if pval < alpha:
        print(f"Heteroskedasticity is present in this model (at the {int(alpha*100)}% significance level)")
    else:
        print("At the 5% significance level there is insufficient evidence to infer that there is heteroskedasticity")

    fig, ax = plt.subplots()
    sns.residplot(x=fitted_model.fittedvalues, y=fitted_model.resid, scatter_kws={'s': 3, 'color': 'black'}, ax=ax)
    ax.set(title='Heteroskedasticity Check', xlabel='Model Predictions for Life Expectancy', ylabel='Residuals')








