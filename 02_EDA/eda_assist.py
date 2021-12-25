import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.impute import KNNImputer
from sklearn.covariance import EllipticEnvelope
from statsmodels.stats.outliers_influence import variance_inflation_factor


def numeric_variables(data):
    return [col for col in data.select_dtypes(include=np.number).columns if col != 'Year']


def summarise(data):

    """
    Similar to df.describe() but has also makes calculations based off inter-quartile rule
    and empirical rule to identify uni-variate outliers
    """

    print("Summary Below")
    print(data.info())
    summary = data.describe()

    # Add variance
    summary.loc['Variance'] = summary.loc['std'] ** 2

    # Add IQR
    summary.loc['IQR'] = summary.loc['75%'] - summary.loc['25%']

    # Use IQR rule to potentially identify univariate outliers
    summary.loc['Q3 + 1.5*IQR (INTER QUARTILE RULE)'] = summary.loc['75%'] + (1.5 * summary.loc['IQR'])
    summary.loc['Q1 - 1.5*IQR (INTER QUARTILE RULE)'] = summary.loc['25%'] - (1.5 * summary.loc['IQR'])

    # Empirical Rule
    summary.loc['mu + 3*sigma (EMPIRICAL RULE)'] = summary.loc['mean'] + 3 * summary.loc['std']
    summary.loc['mu - 3*sigma (EMPIRICAL RULE)'] = summary.loc['mean'] - 3 * summary.loc['std']

    # Display
    return summary


def identify_country_mismatches(train, test, hdi):
    """
    A helper function to see if there are any country mismatches between datasets
    e.g. 'Syria' appearing in the training set but then 'Syrian Arab Republic' appearing in the HDI dataset
    """
    train_countries = set(train.Country)
    test_countries = set(test.Country)
    hdi_countries = set(hdi.Country)

    print("In train not in hdi")
    print(train_countries.difference(hdi_countries))

    print()

    print("Geting everything in test not in hdi")
    print(test_countries.difference(hdi_countries))


def search_substrings(hdi_dataset):
    """
    Direct follow on from the identify_country_mismatches(...) function.
    The aim here is to see why/where the country mismatches are occuring
    """

    correct_names = hdi_dataset['Country'].str.lower().str.contains('russia|cabo|verde|brunei|syria')
    return f"\nIn the HDI dataset these values show as {hdi_dataset.loc[correct_names, 'Country'].unique()}"


def correct_mismatches(train, test):

    train = (
        train.replace('Russian Federation', 'Russia').
        replace('Cabo Verde', 'Cape Verde').
        replace('Brunei Darussalam', 'Brunei').
        replace('Syrian Arab Republic', 'Syria')
    )

    # For the testing set
    test = (
        test.replace('Russian Federation', 'Russia').
        replace('Cabo Verde', 'Cape Verde').
        replace('Brunei Darussalam', 'Brunei').
        replace('Syrian Arab Republic', 'Syria')
    )

    return train, test






def basic_categorical_variable_analysis(data, categorical_feature, fig_s=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(figsize=fig_s, nrows=1, ncols=2)
    vc = data[categorical_feature].value_counts(ascending = False)

    # Bar
    vc.plot(kind='barh',
            title='Count Plot for the "Status" feature',
            ylabel='Count',
            color=['#e6f2ff', '#3399ff'],
            edgecolor=['black'],
            ax=ax1)

    # Pie
    vc.plot(kind='pie',
            autopct='%1.5f%%',
            ax=ax2,
            wedgeprops={'edgecolor': 'black'},
            colors=['#e6f2ff', '#3399ff']);


def show_skewness(data, figsize=(12, 5), style='seaborn-whitegrid'):

    numeric_features = numeric_variables(data)
    new_data = data[numeric_features].copy()

    # skew_summary = new_data.skew(numeric_only=True)
    plt.style.use(style)
    for var in numeric_features:
        # Calculate skew for a particular variable
        skew = np.round(stats.skew(new_data[var], bias=False, nan_policy='omit'), 3)

        fig, (ax1, ax2) = plt.subplots(figsize=figsize, nrows=1, ncols=2)
        ax1.set(title=f'Kernel Density Estimate for {var}')
        ax2.set(title=f'Cumulative Distribution Function for {var}')

        sns.kdeplot(ax=ax1, data=new_data, x=var, color='green', shade=True, label=f'Skewness = {skew}')
        sns.kdeplot(ax=ax2, data=new_data, x=var, color='black', shade=True, cumulative=True)

        # Adjust legend based on skewness for aesthetic purposes
        if skew < 0:
            ax1.legend(loc='upper left', fontsize=12)
        else:
            ax1.legend(loc='upper right', fontsize=12)

        ax1.grid(False)
        ax2.grid(False)


def display_boxplots(data, figsize=(12, 5), style='seaborn-whitegrid'):
    num_vars = numeric_variables(data)
    plt.style.use(style)
    for var in num_vars:
        fig, (ax1, ax2) = plt.subplots(figsize=figsize, nrows=1, ncols=2)
        sns.boxplot(data=data, y=var, color='cyan', ax=ax1)
        sns.boxplot(data=data, x='Status', y=var, palette='summer', hue='Status', ax=ax2)
        ax1.set(title=f"{var} Boxplot")
        ax2.set(title=f"{var} Boxplot ~ Developing countries vs developed")


# ======================================================================================================================
# Bi-variate Analysis
# ======================================================================================================================
def corr_heatmap(data, method='pearson', figsize=(10, 6), style='seaborn-whitegrid'):
    """Plots a correlation heatmap"""
    corr = data.corr(method=method)

    # Lower triangle only since corr(X, Y) = corr(Y, X)
    mask = np.triu(np.ones_like(data.corr(), dtype=bool))

    plt.style.use(style)
    fig, ax = plt.subplots(figsize=figsize)
    x_labels = y_labels = data.corr().columns

    # Ensures that only half of the plot shows. Calling our triu function from NumPy
    sns.heatmap(corr, mask=mask, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels, annot=True, ax=ax)


def plot_corr(data, xvar, target='Life_exp', figsize=(10, 6), style='seaborn-whitegrid'):
    """A function that allows us to understand the correlation between two variables"""

    plt.style.use(style)

    corr = data[xvar].corr(data[target])  # Calculate correlation

    fig, ax = plt.subplots(figsize=figsize)  # Create a figure to put our plots on
    ax.set(title=f"{xvar}:\n Correlation with Life Expectancy = {round(corr, 2)}")
    ax.ticklabel_format(useOffset=False, style='plain')  # No more scientific notation on the labels

    sns.scatterplot(x=xvar, y="Life_exp", data=data, palette=['green', '#42bcf5'], ax=ax, hue='Status')

    # Set up legend
    # Allows our legend to be placed outside the graph
    legend = ax.legend(bbox_to_anchor=[1.2, 1.0], fancybox=True, fontsize='medium')
    legend.get_texts()[0].set_text('Developed')  # Instead of 0 lets just say 'Developing'
    legend.get_texts()[1].set_text('Developing')  # Instead of 1 lets just say 'Developed'

    ax.grid(False)  # No grid lines


def henze_zirkler(data, y='Life_exp', alpha=0.05):
    """
    The Henze-Zirkler test can test for multivariate normality for more than two variables, though this function is just
    concerned with two variables. Returns a pandas dataframe with our results for possible bivariate combinations that
    include the life expectancy variable
    """
    records = []
    for x_var in data.select_dtypes('number').drop(y, axis=1).columns:
        test_data = data[[x_var, y]]
        hz_stat, p_value, normal = pg.multivariate_normality(test_data, alpha=alpha)
        records.append((f"{x_var} and {y}", hz_stat, p_value, normal))

    col_list = ['Variables in Question', 'HZ Statistic', f'P-Value (alpha = {alpha})', 'The Data is Bi-variate Normal?']
    return pd.DataFrame(records, columns=col_list)


def one_class_svm(data, x, y='Life_exp', kernel='rbf', gamma=0.001, nu=0.01, figsize=(10, 6)):
    """
    Runs One Class SVM to return either 1 (not predicted to be an outlier) or -1 (predicted to be an outlier).
    This then gets clearly colour coded in a seaborn scatterplot
    """
    X = data[[x, y]].dropna()
    outlier_detection_model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu).fit(X)
    predictions = outlier_detection_model.predict(X)

    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=x, y=y, data=X, palette=['hotpink', 'black'], hue=predictions, ax=ax)
    legend = ax.legend()
    legend.get_texts()[0].set_text('Potential outliers')
    legend.get_texts()[1].set_text('Not identified as potential outliers')
    plt.grid(False)


# ======================================================================================================================
# Multi-variate Analysis
# ======================================================================================================================

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























