import pytest
import pandas as pd
import numpy as np
import joblib
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scripts.model_preprocess import preprocess

@pytest.fixture
def sample_data():
    """Creates sample user, movie, and rating data."""
    user_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    
    movie_data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    
    rating_data = pd.DataFrame({'rating': np.random.rand(100)})
    
    return user_data, movie_data, rating_data

def test_preprocess(sample_data):
    """Tests the preprocess function."""
    user, movies, y_train = sample_data
    result = preprocess(user, movies, y_train)

    # Ensure preprocess doesn't return None
    assert result is not None, "Preprocess function returned None."

    # Unpack results
    user_train, user_test, movie_train, movie_test, y_train, y_test, scalerUser, scalerItem, scalerTarget = result

    # Check that the splits are correct
    assert user_train.shape[0] == 80, "User train split incorrect."
    assert user_test.shape[0] == 20, "User test split incorrect."
    assert movie_train.shape[0] == 80, "Movie train split incorrect."
    assert movie_test.shape[0] == 20, "Movie test split incorrect."
    assert y_train.shape[0] == 80, "y_train split incorrect."
    assert y_test.shape[0] == 20, "y_test split incorrect."

    # Check if scalers are saved
    assert isinstance(scalerUser, StandardScaler), "User scaler should be a StandardScaler."
    assert isinstance(scalerItem, StandardScaler), "Item scaler should be a StandardScaler."
    assert isinstance(scalerTarget, MinMaxScaler), "Target scaler should be a MinMaxScaler."

    # Check if scaler files exist
    assert joblib.load('./model/scalerUser.pkl'), "User scaler file missing."
    assert joblib.load('./model/scalerItem.pkl'), "Item scaler file missing."
    assert joblib.load('./model/scalerTarget.pkl'), "Target scaler file missing."
