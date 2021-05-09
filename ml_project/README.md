ml_project
==============================

Homework 1. ML in production course 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │
    ├── test               <- Test for source code
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
Для обучения модели можно, находясь в папке проекта воспользоваться командой:
```bash
python src/train_pipeline.py <path to config> <model name>
```
Оригинальный путm до конфиг-фала обучения - `config/train_config.yaml`.
Для параметра `<model name>` доступны следующие значения:
 - LinearSVC
 - SGDClassifier

Пример команды для обучения:
```
python src/train_pipeline.py config/train_config.yaml LinearSVC
```




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
