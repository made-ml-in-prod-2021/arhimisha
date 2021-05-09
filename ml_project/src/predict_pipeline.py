import json
import logging
import sys
import pickle

import click

from src.data import read_data
from src.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def predict_pipeline(path_to_model: str,
                     path_to_transformer: str,
                     path_to_data: str,
                     path_for_save_result: str):
    logger.info(
        f"start predict pipeline with path to model: {path_to_model}, " +
        f"path to transformer: {path_to_transformer}, " +
        f"path to data: {path_to_data}, " +
        f"path for save result: {path_for_save_result}")
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    with open(path_to_transformer, "rb") as f:
        transformer = pickle.load(f)
    logger.info(f"model is {model}")
    logger.info(f"transformer is {transformer}")

    logger.info(f"start transform data with path to data: {path_to_data}")
    data = read_data(path_to_data)
    logger.info(f"data.shape is {data.shape}")

    transformed_data = transformer.transform(data)
    logger.info(f"transformed_data.shape is {transformed_data.shape}")

    logger.info(f"start predict")
    predicts = predict_model(model, transformed_data)
    logger.info(f"predict.shape is {predicts.shape}")

    logger.info(f"save predicts")
    with open(path_for_save_result, "w") as result_file:
        json.dump(predicts.tolist(), result_file)

    return path_for_save_result


@click.command(name="predict_pipeline")
@click.argument("path_to_model")
@click.argument("path_to_transformer")
@click.argument("path_to_data")
@click.argument("path_for_save_result")
def predict_pipeline_command(path_to_model: str,
                             path_to_transformer: str,
                             path_to_data: str,
                             path_for_save_result: str):
    predict_pipeline(path_to_model, path_to_transformer, path_to_data, path_for_save_result)


if __name__ == "__main__":
    predict_pipeline_command()
