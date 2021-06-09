# Airflow ML project

## Run Airflow
Use Docker to run Airflow. Ensure that your Airflow version is  ```2.0.0``` and 
``` apace-airflow-providers-docker``` is installed.
If you run for the first time, initialize Airflow database:
```
docker-compose up init
```
To build necessary docker images and run Airflow use the following command:
```
docker-compose up --build
```
To down the DAGs use the following command:
```
docker-compose down
```
If you modify the DAGs or the scripts inside containers, rebuild container 
```
docker-compose build <service_name>
```
Airflow UI will be available by the link ```localhost:8080``` with login: admin, pwd: admin.
## Test Airflow DAGs
To test the DAGs use the command ```pytest``` from the root directory.
