from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "arhimisha",
    "email": ["miklpostbox@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

prod_model_path = "{{ var.value.model_path }}"

with DAG(
        "prediction",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    wait = FileSensor(
        task_id="model_file_sensor",
        filepath="/opt/airflow/data/model/{{ ds }}/model.pkl",
        poke_interval=20,
    )
    prediction = DockerOperator(
        image="airflow-ml-prediction",
        command=" ".join(["/data/raw/{{ ds }}",
                          prod_model_path,
                          "/data/prediction/{{ ds }}"]),
        task_id="prediction",
        do_xcom_push=False,
        volumes=["D:/Made2020/2_ml_in_prod/homework/airflow_ml_dags/data:/data"]
    )
    wait >> prediction
