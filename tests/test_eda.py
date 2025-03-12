import pandas as pd
import pytest
import logging
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.eda import analysis  # Import your class from the correct module

# Setup a test logger
logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)

@pytest.fixture
def sample_data():
    # Create sample user, movie, y_train, genre, and list datasets
    user_data = pd.DataFrame({'user_id': [1, 2, 3, 1, 2, 4, 5]})
    movie_data = pd.DataFrame({'movie_id': [101, 102, 103, 101, 102, 104, 105], 'year': [2000, 2001, 2002, 2000, 2001, 2003, 2004]})
    y_train = pd.Series([3, 5, 4, 2, 1, 4, 5])  # Ratings
    genre_data = pd.DataFrame({'genre': ['Action', 'Comedy', 'Drama', 'Action', 'Horror', 'Sci-Fi', 'Romance']})
    list_data = pd.DataFrame({'movieId': [101, 102, 103, 104, 105], 'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E']})

    return user_data, movie_data, y_train, genre_data, list_data

def test_bygenre(sample_data):
    user, movie, y_train, genre, list_data = sample_data
    analysis_obj = analysis(user, movie, y_train, genre, list_data, logger)
    assert analysis_obj.bygenre().equals(genre), "Genre classification failed"

def test_no_of_user_mov(sample_data, capsys):
    user, movie, y_train, genre, list_data = sample_data
    analysis_obj = analysis(user, movie, y_train, genre, list_data, logger)
    analysis_obj.no_of_user_mov()
    
    captured = capsys.readouterr()
    assert "Number of users: 5" in captured.out
    assert "Number of movies: 5" in captured.out

def test_most_watched(sample_data):
    user, movie, y_train, genre, list_data = sample_data
    analysis_obj = analysis(user, movie, y_train, genre, list_data, logger)
    result = analysis_obj.most_watched()
    
    assert isinstance(result, pd.DataFrame), "Most watched movies should return a DataFrame"
    assert 'movieId' in result.columns, "Missing 'movieId' column in result"
    assert len(result) <= 10, "Most watched should return at most 10 movies"

def test_years(sample_data):
    user, movie, y_train, genre, list_data = sample_data
    analysis_obj = analysis(user, movie, y_train, genre, list_data, logger)
    try:
        analysis_obj.years()
    except Exception:
        pytest.fail("Years plot failed")

def test_most_active_user(sample_data):
    user, movie, y_train, genre, list_data = sample_data
    analysis_obj = analysis(user, movie, y_train, genre, list_data, logger)
    try:
       analysis_obj.most_active_user()
    
    except Exception:
        pytest.fail("Most active users plot failed")

