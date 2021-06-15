from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable

default_args = {
    "owner": "arhimisha",
    "email": ["miklpostbox@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

min_score = Variable.get("min_score")
prod_model_path = Variable.get("model_path")

with DAG(
        "training_model",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    wait_data = FileSensor(
        task_id="data_file_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",
        poke_interval=20,
    )
    wait_target = FileSensor(
        task_id="target_file_sensor",
        filepath="/opt/airflow/data/raw/{{ ds }}/target.csv",
        poke_interval=20,
    )
    prepare_data = DockerOperator(
        image="airflow-ml-prepare-data",
        command="/data/raw/{{ ds }} /data/prepared/{{ ds }}",
        task_id="prepare_data",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    check_data = DockerOperator(
        image="airflow-ml-check-data",
        command="/data/prepared/{{ ds }}",
        task_id="check_data",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    split_data = DockerOperator(
        image="airflow-ml-split-data",
        command="/data/prepared/{{ ds }}",
        task_id="split_data",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    training_model = DockerOperator(
        image="airflow-ml-training-model",
        command="/data/prepared/{{ ds }} /data/model/{{ ds }} ",
        task_id="training_model",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    validation_model = DockerOperator(
        image="airflow-ml-validation-model",
        command=" ".join(["/data/prepared/{{ ds }}",
                          "/data/model/{{ ds }}",
                          str(min_score),
                          prod_model_path]),
        task_id="validation_model",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    [wait_data, wait_target] >> prepare_data
    prepare_data >> [check_data, split_data] >> training_model >> validation_model
