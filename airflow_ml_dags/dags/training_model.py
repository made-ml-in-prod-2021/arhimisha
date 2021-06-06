from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "arhimisha",
    "email": ["miklpostbox@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "prepare_data",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    prepare_data = DockerOperator(
        image="airflow-ml-prepare-data",
        command="/data/raw/{{ ds }} /data/prepared/{{ ds }}",
        task_id="prepare-data",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    check_data = DockerOperator(
        image="airflow-ml-check-data",
        command="/data/prepared/{{ ds }}",
        task_id="check-data",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    prepare_data >> check_data
