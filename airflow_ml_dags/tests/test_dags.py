import sys
import pytest
from airflow.models import DagBag

sys.path.append("dags")


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


def test_dag_bag_import(dag_bag):
    assert dag_bag.dags is not None
    assert dag_bag.import_errors == {}
    assert "generate_data" in dag_bag.dags
    assert "training_model" in dag_bag.dags
    assert "prediction" in dag_bag.dags


def test_dag_generate_data(dag_bag):
    flow = {
        "generate-data": [],
    }
    dag = dag_bag.dags["generate_data"]
    for name, task in dag.task_dict.items():
        assert set(flow[name]) == task.downstream_task_ids


def test_dag_training_model(dag_bag):
    structure = {
        "data_file_sensor": ["prepare_data"],
        "target_file_sensor": ["prepare_data"],
        "prepare_data": ["split_data", "check_data"],
        "check_data": ["training_model"],
        "split_data": ["training_model"],
        "training_model": ["validation_model"],
        "validation_model": [],
    }
    dag = dag_bag.dags["training_model"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids


def test_dag_prediction(dag_bag):
    structure = {
        "model_file_sensor": ["prediction"],
        "prediction": [],
    }
    dag = dag_bag.dags["prediction"]
    for name, task in dag.task_dict.items():
        assert set(structure[name]) == task.downstream_task_ids
