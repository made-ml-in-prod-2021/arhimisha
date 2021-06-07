import os
import click
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("training_model")
@click.argument("data_dir")
@click.argument("model_dir")
def training_model(data_dir: str, model_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "data_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "target_train.csv"))
    model = LogisticRegression()
    model.fit(X_train, y_train)
    model_path = os.path.join(model_dir, "model.pkl")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    training_model()
