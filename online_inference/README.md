# Homework 2 Docker and REST API


Команда для осуществления предсказания:
`
python make_request.py http://localhost/ data/test_data_for_predict.csv result.json
`

Команды Docker
`
docker build -t arhimisha/ml_in_prod_task_2:v1 .
docker run -p 80:80 arhimisha/ml_in_prod_task_2:v1
docker push arhimisha/ml_in_prod_task_2:v1
`
