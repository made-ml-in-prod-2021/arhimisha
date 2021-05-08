import os
from py._path.local import LocalPath

from src.train_pipeline import train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    ModelParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_path: str,
    feature_params: FeatureParams,
    splitting_params: SplittingParams,
    model_params: ModelParams
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
        model_params=[model_params]
    )
    real_model_path, real_transformer_path, metrics = train_pipeline(params, model_params.model_name)
    for score in metrics.values():
        assert score > 0.8
    assert os.path.exists(real_model_path)
    assert os.path.exists(real_transformer_path)
    assert os.path.exists(params.metrics_path)
