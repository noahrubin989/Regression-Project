import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV


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
    linear_model = (isinstance(model, LinearRegression)
                    or isinstance(model, Ridge)
                    or isinstance(model, Lasso)
                    or isinstance(model, ElasticNet))
    
    if linear_model:
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

