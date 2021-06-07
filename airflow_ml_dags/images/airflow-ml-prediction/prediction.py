import os
import click
import pickle
import pandas as pd
from airflow.models import Variable


@click.command("prediction")
@click.argument("data_dir")
@click.argument("model_dir")
@click.argument("output_dir")
def prediction(data_dir: str, model_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(data_dir, "data.csv"))

    # Variables not work
    # model_path = Variable.get("model_path")
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(y_pred).to_csv(os.path.join(output_dir, "prediction.csv"), index=False)


if __name__ == "__main__":
    prediction()
