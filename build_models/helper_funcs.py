import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

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
    design_matrix = sm.add_constant(data=X_train_sm)
    return design_matrix, X_test_sm


# ======================================================================================================================
#                                                   OLS Model Diagnostics
# ======================================================================================================================

def adjusted_r2(model, x_test, y_test):
    """Calculates adjusted R^2"""
    n = len(y_test)
    k = len(x_test.columns)
    r2 = model.score(x_test, y_test)
    numerator = (1 - r2) * (n-1)
    denominator = n - k - 1
    return 1 - (numerator / denominator)


def test_heteroskedasticity(fitted_model, alpha=0.05):
    """
    :param fitted_model: An already fitted model ready to plot out and run tests on
    :return: A scatterplot and printed results of the Breusch Pagan test
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


def check_residual_normality(fitted_model, residual_data):
    """A function that returns a ks test results, qq plot and a density graph for model residuals"""

    # Calculate test statistic and p_value for the KS test
    d_statistic, p_value = stats.kstest(fitted_model.resid, 'norm')

    print("Kolmogorov Smirnov test results:")
    print("-" * 35)
    print(f"D-statistic: {d_statistic}")
    print(f"p-value: {p_value}")

    if p_value <= 0.05:
        print("At the 5% significance level, the data does not follow a normal distribution")
    else:
        print("At the 5% significance level, the data does indeed follow a normal distribution")

    fig, (ax1, ax2) = plt.subplots(nrows=1,
                                   ncols=2,
                                   figsize=(12, 5))

    # Calculate skewness
    skewness = stats.skew(residual_data, bias=False)

    # This is for the QQ plot
    sm.qqplot(residual_data, ax=ax1, fit=True, line="45")

    # Density curve
    sns.kdeplot(x=residual_data,
                color='g',
                shade=True,
                ax=ax2,
                label=f"Skewness = {round(skewness, 2)}")

    # Graph display settings
    ax1.set(title='QQ plot')
    ax1.grid(False)

    ax2.set(title='Density curve');
    ax2.grid(False)
    ax2.legend(loc='upper left', fontsize=10);


def display_vif(feature_matrix, threshold=5):
    """
    This function calculates the variance inflation factor for
    our x variables (covariates), returning a graph.

    A variance inflation factor over 5 is a sign
    that multicoliniarity exists.

    Simply pass in all of your covariates for the feature_marix argument as a dataframe
    """

    # We have to manually add the constant term
    x_new = sm.tools.add_constant(feature_matrix)

    # Calculating VIF
    vif = pd.DataFrame({'Variable': x_new.columns,
                        'VIF': [variance_inflation_factor(x_new.values, exog_idx=i) for i in range(x_new.shape[1])]})

    fig, ax = plt.subplots(figsize=(12, 6))

    # Our VIF calculation has the VIF of the constant term but we typically don't consider it
    # ...(hence the iloc[1:, :] on the line below)
    vif_plot = vif.iloc[1:, :].plot(ax=ax,
                                    kind='bar',
                                    x='Variable',
                                    y='VIF',
                                    color='blue',
                                    edgecolor='black',
                                    title='VIF for Each Predictor Variable',
                                    ylabel='VIF')

    ax.axhline(y=threshold, color='r', linestyle='-.', label=f'VIF Threshold = {threshold}')
    ax.legend(fontsize='large', loc='best')
    ax.grid(False)

    return vif_plot


def make_prediction(input_list, pipeline, x_test):
    prediction_input = pd.DataFrame([input_list], columns=x_test.columns)
    prediction_output = pipeline.predict(prediction_input)[0]
    return prediction_output


def display_regression_metrics(y_actual, y_pred):
    r2 = r2_score(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    print(f"R^2 = {r2}\nMean Squared Error = {mse}\nRoot Mean Squared Error = {rmse}\nMean Absolute Error = {mae}")
    return r2, mse, rmse, mae









