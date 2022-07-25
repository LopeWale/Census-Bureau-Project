import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import logging

#get the data from the dir and save it as a pandas' dataframe
def get_data():
    dataset = pd.read_csv("data/census.csv")
    X = pd.DataFrame(dataset)
    cleaned_column_names = (X.columns
                            .str.strip()
                            .str.replace('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))', r'_\1')
                            .str.lower()
                            .str.replace('[ _-]+', '_')
                            .str.replace('[}{)(><.!?\\\\:;,-]', ''))
    X.columns = cleaned_column_names

    X["education"] = X["education"].replace(dict.fromkeys(['9th', '10th', '11th', '12th'], 'Some-HS'))
    X.to_csv('./data/clean_data.csv')
    return X

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # get the categorical columns to a list
    for cat in X.select_dtypes(['object']).columns:
        X[cat] = X[cat].str.strip()

        """replace the columns with '?' which are all in the categorical columns with 'unknown'
        keeping this rows because there is not more than 2 values with '? per row, 
        and believe rows with '?' still has valuable information for the model"""

        X[cat] = X[cat].str.replace('?', 'unknown', regex=False)
        categorical_features.append(cat)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError as err:
            raise err

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
