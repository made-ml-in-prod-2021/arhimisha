ml_project
==============================

Homework 1. ML in production course 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── external       <- Data from third party sources.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks with EDA.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    └── test               <- Test for source code

--------
Для обучения модели можно, находясь в папке проекта воспользоваться командой:
```shell
python ml_project/src/train_pipeline.py <path to config> <model name>
```
Оригинальный путь до конфиг-файла обучения - `ml_project/config/train_config.yaml`.
Для параметра `<model name>` доступны следующие значения:
 - LinearSVC
 - SGDClassifier

Пример команды для обучения:
```shell
python ml_project/src/train_pipeline.py ml_project/config/train_config.yaml LinearSVC
```
После обучения, полученная модель и трансформер данных сохраняются в указанный в конфигурационном файле путях.
Их можно использовать для предсказания результатов для аналогичных данных с помощью другой команды, находясь в папке проекта:
```shell
python ml_project/src/predict_pipeline.py <path to model> <path to transformer> <path to data> <path for save result>
```
 - `<path to model>` - Путь до сохраненной модели после обучения
 - `<path to transformer>` - Путm до сохраненного трансформера данных после обучения
 - `<path to data>` - Путь до данных в формате .csv аналогично файлу `ml_project/tests/test_data_for_predict.csv`
 - `<path for save result>` - Путь для сохранения результата работы модели.

Пример команды:
```shell
python ml_project/src/predict_pipeline.py ml_project/models/model.pkl ml_project/models/transformer.pkl ml_project/tests/test_data_for_predict.csv result.json
```


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
