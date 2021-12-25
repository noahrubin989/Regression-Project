import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def compare_status(data, figsize=(12, 5), style='seaborn-whitegrid'):
    """
    Description: Compares developed world with the developing world in year-by-year fashion for our continuous variables
    Input: A dataframe (required), a figure size for each plot and a style (all optional)
    Output: A variety of graphs comparing the developed and developing world
    """
    plt.style.use(style)

    developed = data[data.Status == 'Developed'].copy()
    developing = data[data.Status == 'Developing'].copy()

    # All columns except year
    for col in data.select_dtypes(include=np.number).columns[1:]:
        # Create and set up a new figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=f'Mean {col}: Developed vs Developing', ylabel=col)

        # Developed Countries
        developed.groupby('Year')[col].mean().plot(kind='line', color='#cc00ff', label='Developed', ax=ax)

        # Developing Countries
        developing.groupby('Year')[col].mean().plot(kind='line', color='#6699ff', label='Developing', ax=ax)
        ax.legend()
        ax.grid(False)


def obtain_largest_pct_growth(data):
    """
    Description:
    A helper function to create a dataset that stores relative gdp growth (in order)

    - Input: An existing dataframe
    - Output: A new dataframe storing
    """
    records = []
    for country in data.Country.unique():
        initial = data.loc[(data.Country == country) & (data.Year == data.Year.min()), 'GDP_cap'].values[0]
        final = data.loc[(data.Country == country) & (data.Year == data.Year.max()), 'GDP_cap'].values[0]
        pct_change = ((final - initial) / initial) * 100
        records.append((country, pct_change))

    pct_changes = pd.DataFrame(records, columns=['Country', '% Change in GDP'])
    pct_changes.sort_values(ascending=False, by='% Change in GDP', inplace=True)
    return pct_changes


def plot_gdp_growth(dataset, top_n_bar, top_n_line, style='seaborn-whitegrid'):
    plt.style.use(style)
    df = pd.read_csv('../datasets/OriginalDataset.csv')

    data = obtain_largest_pct_growth(dataset)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.set(title=f'As a Bar Plot (Top {top_n_bar})')
    ax1.tick_params(axis='x', rotation=90)
    ax2.set(title=f'As a Line Plot (Top {top_n_line})')
    ax2.grid(False)

    sns.barplot(x=data.Country[:top_n_bar], y=data['% Change in GDP'][:top_n_bar], ax=ax1)
    lineplot_countries = data.Country[:top_n_line]
    sns.lineplot(data=df[df.Country.isin(lineplot_countries)], x='Year', y='GDP_cap', hue='Country', ax=ax2)
    plt.xticks(np.arange(2000, 2020, 4))


def low_life_exp(data, value=50, figsize=(12, 5)):
    """
    Input:
    - A pandas dataframe
    - A value for what you can use to find all countries where life expectancy at some stage fell below that number
    - Values over 50 not accepted
    - You can also pass in the size of your figure as a tuple

    Output:
    - Two graphs:
    1. A bar plot with the 10 countries with lowest average life expectancy according to our dataset
    2. All countries where life expectancy was at some stage below a specified value that ther user enters
    """
    assert value <= 50, "Values over 50 not accepted"
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Graph 1
    # ==================================================================================================================
    ax1.set(title="Countries with Lowest AVG Life Expectancy")
    gb = data.groupby('Country').Life_exp.mean().sort_values()[:10]
    sns.barplot(x=gb.index, y=gb.values, ax=ax1, palette='Blues', edgecolor='black')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)  # Rotate x tick labels 90 degrees

    # Graph 2
    # ==================================================================================================================
    life_exp_low = data[data.Life_exp <= value].copy()
    countries = life_exp_low.Country.unique()
    ax2.set(title=f"Countries where life expectancy has been less than {value}")
    sns.lineplot(x='Year',
                 y='Life_exp',
                 hue='Country',
                 palette='tab10',
                 ax=ax2,
                 data=data[data['Country'].isin(countries)])

    plt.xticks(np.arange(2000, 2020, 4))  # Have x ticks increment by 4
    plt.legend(bbox_to_anchor=[1, 1])  # Place legend outside the graph
    plt.grid(False)


def mean_median_comparison(data, aggregation_variable, categorical_feature='Status', figsize=(10, 6), style='seaborn'):
    """
    Input:
    - A pandas dataframe
    - A single variable to aggregate on when running group by
    - The variable to group in terms of ('Status' by default)
    - A figure size for the graph

    Output:
    - A basic plot comparing developed and developing countries (using mean and median as aggregation functions)
    """
    plt.style.use(style)
    gb = data.groupby(by=categorical_feature)[aggregation_variable].agg(['mean', 'median'])
    gb.plot(kind='bar',
            figsize=figsize,
            title=f'{aggregation_variable} Comparison',
            edgecolor='black',
            ylabel=aggregation_variable,
            cmap='Set2')


def top_five_bottom_five(data, aggregation_variable, style='seaborn-whitegrid', palette='tab10'):
    """
    Description:
    The data will be grouped by country and we can specify the [aggregation_variable] parameter that we want to do the
    mean aggregation function on. Shows a graph of the top and bottom five performing countries

    Input: A dataframe and a variable to aggregate on
    Output: A plot displaying the top and bottom 5 countries based on the aggregation variable passed in
    """

    # ========================================== Set up figure =========================================================
    plt.style.use(style)
    fig, (ax1, ax2) = plt.subplots(constrained_layout=True, nrows=2, sharex=True, figsize=(12, 10))
    fig.suptitle(f'Top/Bottom: {aggregation_variable}')

    # ========================= Create dataset with best and worst performing countries ===============================
    worst = data.groupby('Country')[aggregation_variable].mean().sort_values(ascending=True)[:5]
    data_worst = data[data.Country.isin(worst.index)]
    best = data.groupby('Country')[aggregation_variable].mean().sort_values(ascending=False)[:5]
    data_best = data[data.Country.isin(best.index)]

    # ====================================== Create two separate line plots ============================================
    sns.lineplot(x=data_worst.Year, y=aggregation_variable, hue='Country', palette=palette, data=data_worst, ax=ax1)
    sns.lineplot(x=data_best.Year, y=aggregation_variable, hue='Country', palette=palette, data=data_best, ax=ax2)

    # ============================================= Display settings ===================================================
    ax1.set(xlim=(data_best.Year.min() - 1, data_best.Year.max() + 1))
    ax1.ticklabel_format(useOffset=False, style='plain')
    plt.xticks(np.arange(2000, 2020, 2))
    ax2.set(xlim=(data_worst.Year.min() - 1, data_best.Year.max() + 1))
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax1.grid(False)
    ax2.grid(False)


def obtain_largest_pct_growth(data):
    """
    Description:
    A helper function to create a dataset that stores relative gdp growth (in order)

    - Input: An existing dataframe
    - Output: A new dataframe storing
    """
    records = []
    for country in data.Country.unique():
        initial = data.loc[(data.Country == country) & (data.Year == data.Year.min()), 'GDP_cap'].values[0]
        final = data.loc[(data.Country == country) & (data.Year == data.Year.max()), 'GDP_cap'].values[0]
        pct_change = ((final - initial) / initial) * 100
        records.append((country, pct_change))

    pct_changes = pd.DataFrame(records, columns=['Country', '% Change in GDP'])
    pct_changes.sort_values(ascending=False, by='% Change in GDP', inplace=True)
    return pct_changes

