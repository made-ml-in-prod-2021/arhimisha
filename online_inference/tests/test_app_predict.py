import pandas as pd
from fastapi.testclient import TestClient
from app_predict import app, XInput, YResponse


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "it is entry point of our predictor"


def test_predict_one(dataset_for_predict_path: str):
    with TestClient(app) as client:
        data = pd.read_csv(dataset_for_predict_path)
        data_for_predict = XInput(data=data.loc[[0]].values.tolist(), features=data.columns.tolist())
        response = client.get("/predict/", json=data_for_predict.dict())
        assert response.status_code == 200
        assert len(response.json()) == 1
        predict = YResponse(**response.json()[0])
        assert predict.predict in [0, 1]


def test_predict_batch(dataset_for_predict_path: str):
    with TestClient(app) as client:
        data = pd.read_csv(dataset_for_predict_path)
        data_for_predict = XInput(data=data.values.tolist(), features=data.columns.tolist())
        response = client.get("/predict/", json=data_for_predict.dict())
        assert response.status_code == 200
        assert len(response.json()) == data.shape[0]
        for elem in response.json():
            predict = YResponse(**elem)
            assert predict.predict in [0, 1]
