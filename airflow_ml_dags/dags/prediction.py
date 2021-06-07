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
        "prediction",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    prediction = DockerOperator(
        image="airflow-ml-prediction",
        command="/data/raw/{{ ds }} /data/model/{{ ds }} /data/prediction/{{ ds }}",
        task_id="prediction",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
