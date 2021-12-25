import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from statsmodels.stats import diagnostic as diag

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
#                                                   Diagnostics
# ======================================================================================================================
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

    print(f"D: {d_statistic}\n")

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
    ax2.legend(loc='upper left', fontsize=14);





