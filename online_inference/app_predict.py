import sys
import logging
import os
import pickle
from typing import List, Union, Optional, NoReturn
import time

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from src.entities.features_info import FeaturesInfo, read_features_info


START_DELAY_SECONDS = 30
DEAD_DELAY_SECONDS = 60

logger = logging.getLogger(__name__)
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
features_info: Optional[FeaturesInfo] = None


def valid_data(
        data: List[List[Union[float, str]]],
        features: List[str],
        features_info: FeaturesInfo
) -> NoReturn:
    logger.info(f"check data")
    if not features_info.features_number == len(features):
        details = f"Wrong number of features: {features_info.features_number} != {len(features)}\n" \
                  f"expected_features={features_info.feature_names}\n" \
                  f"features={features}"
        logger.error(details)
        raise HTTPException(status_code=400, detail=details)
    for x in data:
        if not len(x) == len(features):
            details = f"Wrong number of features in data: {len(x)} != {len(features)}"
            logger.error(details)
            raise HTTPException(status_code=400, detail=details)
    for expected_feature, given_feature in zip(features_info.feature_names, features):
        if not expected_feature == given_feature:
            details = f"Wrong names of features:\n" \
                      f"expected_feature={expected_feature}\n" \
                      f"given_feature={given_feature}"
            logger.error(details)
            raise HTTPException(status_code=400, detail=details)


def make_predict(
        data: List[List[Union[float, str]]],
        features: List[str],
        model: SklearnModel,
        transformer: Pipeline,
        features_info: FeaturesInfo
) -> List[YResponse]:
    valid_data(data, features, features_info)

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
start_time = time.time()


@app.get("/")
async def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    global transformer
    global features_info

    # delay
    time.sleep(START_DELAY_SECONDS)

    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        model_path = "models/model.pkl"

    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    if transformer_path is None:
        transformer_path = "models/transformer.pkl"

    features_info_path = os.getenv("PATH_TO_FEATURE_INFO")
    if features_info_path == None:
        features_info_path = "models/features_info.yaml"

    model = load_object(model_path)
    transformer = load_object(transformer_path)
    features_info = read_features_info(features_info_path)


@app.get("/predict/", response_model=List[YResponse])
async def predict(request: XInput):
    return make_predict(request.data, request.features, model, transformer, features_info)


@app.get("/healthz")
async def healthz() -> bool:
    work_time = time.time() - start_time
    if model is not None and work_time < DEAD_DELAY_SECONDS:
        response = PlainTextResponse("OK", status_code=200)
    else:
        response = PlainTextResponse("Bad", status_code=400)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


if __name__ == "__main__":
    uvicorn.run("app_predict:app", host="0.0.0.0", port=os.getenv("PORT", 80))
