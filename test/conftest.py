import pytest
import yaml
import pandas as pd
from fastapi.testclient import TestClient
from main import app
from starter.ml.data import get_data, process_data


@pytest.fixture
def raw_data():
    """
    Get dataset
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    return df


@pytest.fixture
def clean_data(raw_data):
    """
    Get dataset
    """
    df = get_data()
    return df


@pytest.fixture
def cat_features():
    """
    Get dataset
    """
    return process_data.categorical_features


@pytest.fixture
def train_data(get_data):
    """
    Get dataset
    """
    df = get_data()
    df.drop('salary', axis=1)
    return df


@pytest.fixture
def test_data(get_data):
    """
    Get dataset
    """
    df = get_data()
    df = df['salary']
    return df


@pytest.fixture
def inference_data_low():
    data_dict = {'age': 29,
                 'workclass': 'Private',
                 'fnlgt': 34516,
                 'education': 'Some-HS',
                 'marital-status': 'Married',
                 'occupation': 'Own-child',
                 'relationship': 'Husband',
                 'race': 'Black',
                 'sex': 'Male',
                 'hours-per-week': 50,
                 'native-country': 'United-States'
                 }
    df = pd.DataFrame(data=data_dict.values(),
                      index=data_dict.keys()).T
    return df


@pytest.fixture
def inference_data_high():
    data_dict = {'age': 33,
                 'workclass': 'Private',
                 'fnlgt': 257302,
                 'education': 'Assoc-acdm',
                 'marital-status': 'Married-civ-spouse',
                 'occupation': 'Tech-support',
                 'relationship': 'Wife',
                 'race': 'White',
                 'sex': 'Female',
                 'hours-per-week': 38,
                 'native-country': 'United-States'
                 }
    df = pd.DataFrame(data=data_dict.values(),
                      index=data_dict.keys()).T
    return df


@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client
