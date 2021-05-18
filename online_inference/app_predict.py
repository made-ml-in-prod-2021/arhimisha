import sys
import logging
import os
import pickle
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

SklearnModel = Union[LinearSVC, SGDClassifier]
LoadedObjects = Union[SklearnModel, Pipeline]


def load_object(path: str) -> LoadedObjects:
    with open(path, "rb") as f:
        return pickle.load(f)


class XInput(BaseModel):
    data: List[List[Union[float, str]]]
    features: List[str]


class YResponse(BaseModel):
    predict: int


model: Optional[SklearnModel] = None
transformer: Optional[Pipeline] = None


def make_predict(
        data: List[List[Union[float, str]]],
        features: List[str],
        model: SklearnModel,
        transformer: Pipeline
) -> List[YResponse]:
    # todo проверка колонок?
    logger.info(f"start predict")
    logger.info(f"model is {model}")
    logger.info(f"transformer is {transformer}")
    data = pd.DataFrame(data, columns=features)
    logger.info(f"data.shape is {data.shape}")
    logger.info(f"start transform data")
    transformed_data = transformer.transform(data)
    logger.info(f"transformed_data.shape is {transformed_data.shape}")
    logger.info(f"start predict")
    predicts = model.predict(transformed_data)
    logger.info(f"predict.shape is {predicts.shape}")
    return [YResponse(predict=p) for p in list(predicts)]


app = FastAPI()


@app.get("/")
async def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    global transformer
    # todo correct PATH_TO_MODEL
    model_path = os.getenv("PATH_TO_MODEL")
    model_path = "models/model.pkl"
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    # todo correct transformer_path
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    transformer_path = "models/transformer.pkl"
    if transformer_path is None:
        err = f"transformer_path {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)
    transformer = load_object(transformer_path)


@app.get("/predict/", response_model=List[YResponse])
async def predict(request: XInput):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app_predict:app", host="0.0.0.0", port=os.getenv("PORT", 80))
