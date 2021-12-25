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


def create_pipeline(model):
    """
    :param model: A model object to pass in e.g. LinearRegression()
    :return: An sklearn pipeline object with all preprocessing steps added, ready to pass into
    cross_val_score or RandomisedSearchCV or GridSearchCV

    Preprocessing steps include:
    # Steps include:
    # - One hot encoding to our categorical feature
    # - KNN Imputation
    # - Initialise model object
    """
    imp = KNNImputer()
    if isinstance(model, LinearRegression):
        # Apply a log transform to the GDP per capita feature
        numeric_preprocessing = Pipeline(steps=[('logger', FunctionTransformer(np.log))])
    else:
        # Don't log any of the features
        numeric_preprocessing = Pipeline(steps=[('identity', FunctionTransformer(func=None))])

    categorical_preprocessing = Pipeline(steps=[('ohe', OneHotEncoder(drop='first'))])

    # Add all the transformers to a list to eventually pass into ColumnTransformer
    transformers = list()
    transformers.append(('numeric', numeric_preprocessing, ['GDP_cap']))
    transformers.append(('categorical', categorical_preprocessing, ['Status']))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    steps = [('preprocessor', preprocessor), ('imputation', imp), ('model', model)]
    pipeline_object = Pipeline(steps=steps)
    return pipeline_object
