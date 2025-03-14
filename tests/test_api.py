import pytest
import requests
import time
import subprocess

API_URL = "http://127.0.0.1:5000/predict"

# Sample test input
TEST_USER_INPUT = {
    "new_user_id": 5000,
    "new_rating_count": 3,
    "new_rating_ave": 0.0,
    "new_action": 0.0,
    "new_adventure": 0.0,
    "new_animation": 0.0,
    "new_childrens": 0.0,
    "new_comedy": 5.0,
    "new_crime": 0.0,
    "new_documentary": 0.0,
    "new_drama": 0.0,
    "new_fantasy": 0.0,
    "new_horror": 0.0,
    "new_mystery": 0.0,
    "new_romance": 0.0,
    "new_scifi": 0.0,
    "new_thriller": 0.0
}

@pytest.fixture(scope="module", autouse=True)
def start_api():
    """ Start the Flask API before running tests """
    print("\nStarting Flask API...")
    api_process = subprocess.Popen(["python3", "api/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(10)  # Give the API time to start

    yield

    print("\nStopping Flask API...")
    api_process.terminate()

def test_predict_endpoint():
    """ Test if /predict returns a valid response """
    response = requests.post(API_URL, json=TEST_USER_INPUT)
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    assert "error" not in data, f"API returned an error: {data}"
    
    # Fix: Extract list from 'predictions' key
    assert "predictions" in data, "Response should contain a 'predictions' key"
    assert isinstance(data["predictions"], list), "Response['predictions'] should be a list of recommendations"

if __name__ == "__main__":
    pytest.main()
