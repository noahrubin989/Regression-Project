import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def country_comparisons(data, figsize=(12, 5), style='seaborn-whitegrid', palette='tab10'):
    """
    Takes user input on what countries to plot and what variable to consider
    """

    plt.style.use(style)

    print("Countries to choose from:\n")
    print(list(data.Country.sort_values().unique()))
    countries = input("\nCountries to compare against each other (separate by commas): ")
    countries = countries.split(',')
    countries = map(lambda x: x.strip().title(), countries)  # list of countries

    # Ask for the variable they want to include
    print("\nVariables to choose from:\n")
    numeric_variables = [col for col in data.select_dtypes(include=np.number).columns if col != 'Year']
    print(numeric_variables)

    to_consider = input('Enter variable (above) to plot `Year` against (choose one variable): ').strip()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title=to_consider, xticks=np.arange(2000, 2020, 2))
    sns.lineplot(x='Year',
                 y=to_consider,
                 hue='Country',
                 palette=palette,
                 ax=ax,
                 data=data[data.Country.isin(countries)])
    plt.grid(False)
