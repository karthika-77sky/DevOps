from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Iris Classification API is running!"


def test_prediction():
    response = client.post(
        "/predict",
        json=[5.1, 3.5, 1.4, 0.2]
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
