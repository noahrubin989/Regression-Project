import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def process_inconsistent_data(data):
    """
    Description:
    Will ensure that  'Aus', 'aus' and '    aus   ' are all treated the same

    Input: a pandas dataframe
    Output: A pandas dataframe with fixed case and whitespace issues
    """
    data.columns = data.columns.str.strip()
    return data.applymap(lambda x: x.title().strip() if type(x) == str else x)


def display_missing_data(data, figsize=(10, 6), bar_colour='red', edgecolour='black'):
    """
    Input: a pandas dataframe and some matplotlib graph settings
    Result is a matplotlib graph displaying our missing data
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax1.set(title='Number of missing values')
    ax2.set(title='Thin lines represent missing data')
    fig.suptitle('Summary of Missing Data')
    data.isnull().sum().plot(kind='bar',
                             color=[bar_colour],
                             edgecolor=edgecolour,
                             grid=False,
                             ax=ax1,
                             ylabel="Total of Values Missing")

    sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='winter', ax=ax2);



















