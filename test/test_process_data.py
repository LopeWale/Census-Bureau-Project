import os
from starter.ml.data import data_cleaning_stage


def test_clean_data():
    assert os.path.isfile('./data/clean_data.csv')
