import os
from typing import List
from py._path.local import LocalPath

from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    ModelParams,
)


def test_train_e2e(
        tmpdir: LocalPath,
        dataset_path: str,
        dataset_for_persict_path: str,
        feature_params: FeatureParams,
        splitting_params: SplittingParams,
        models_params: List[ModelParams]
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_output_transformer_path = tmpdir.join("transformer.pkl")
    expected_metrics_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=dataset_path,
        output_model_path=expected_output_model_path,
        output_transformer_path=expected_output_transformer_path,
        metrics_path=expected_metrics_path,
        splitting_params=splitting_params,
        feature_params=feature_params,
        models_params=models_params
    )
    for model_params in models_params:
        real_model_path, real_transformer_path, metrics = train_pipeline(params, model_params.model_name)
        for score in metrics.values():
            assert score > 0.7
        assert os.path.exists(real_model_path)
        assert os.path.exists(real_transformer_path)
        assert os.path.exists(params.metrics_path)

        result_path = tmpdir.join("result.json")
        expected_result_path = predict_pipeline(
            real_model_path,
            real_transformer_path,
            dataset_for_persict_path,
            result_path
        )
        assert os.path.exists(expected_result_path)
