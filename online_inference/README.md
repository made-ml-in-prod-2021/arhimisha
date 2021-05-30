# Homework 2 Docker and REST API

В данном заданием выполнена реализация модели предсказания в форме REST-сервиса доступного для запуска в виде docker-файла.

Пример команды формирования докер образа:
```shell
docker build -t arhimisha/ml_in_prod_task_2:v1 .
```
После формирования образа его можно запустить следующей командой:
```shell
docker run -p 80:80 arhimisha/ml_in_prod_task_2:v1
```
Сервис станет доступен по адресу `http://localhost/`


Для осуществления предсказания можно воспользоваться скриптом `make_request.py`.
Запустить скрипт можно с помощью команды:
```shell
python make_request.py server_address path_to_data path_for_save_result
```
 - server_address - адрес запущенного сервера,
 - path_to_data - путь до данных, по которым будут строиться предсказания,
   пример данных представлен в файле `data/test_data_for_predict.csv`,
 - path_for_save_result - путь по которому необходимо сохранить результаты.

Пример команды:
```shell
python make_request.py http://localhost/ data/test_data_for_predict.csv result.json
```

Полученный docker-образ опубликован на docker-hub: 
[https://hub.docker.com/repository/docker/arhimisha/ml_in_prod_task_2](https://hub.docker.com/repository/docker/arhimisha/ml_in_prod_task_2)

Для локального запуска обзара можно воспользоваться командами:
```shell
docker pull arhimisha/ml_in_prod_task_2:v1
docker run -p 80:80 arhimisha/ml_in_prod_task_2:v1
```
