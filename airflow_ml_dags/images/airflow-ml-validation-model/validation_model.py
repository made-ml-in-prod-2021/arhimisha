import os
import click
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from airflow.models import Variable


@click.command("validation_model")
@click.argument("data_dir")
@click.argument("model_dir")
def validation_model(data_dir: str, model_dir: str):
    X_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "target_test.csv"))

    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    with open(os.path.join(model_dir, "score.pkl"), "wb") as f:
        pickle.dump(score, f)

    # Variables not work
    # min_score = Variable.get("min_score")
    min_score = 0.8
    if score > min_score:
        pass
        # Variables not work
        # Variable.set("model_path", model_path)


if __name__ == "__main__":
    validation_model()
