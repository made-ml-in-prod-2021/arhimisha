import os
import json
import click
import pickle
import shutil
import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validation_model")
@click.argument("data_dir")
@click.argument("model_dir")
@click.argument("min_score")
@click.argument("prod_model_path")
def validation_model(data_dir: str,
                     model_dir: str,
                     min_score: str,
                     prod_model_path: str
                     ):
    X_test = pd.read_csv(os.path.join(data_dir, "data_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "target_test.csv"))

    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    with open(os.path.join(model_dir, "score.json"), "w") as f:
        json.dump(score, f)

    if score > float(min_score) or not os.path.exists(prod_model_path):
        shutil.copy2(model_path, prod_model_path)


if __name__ == "__main__":
    validation_model()
