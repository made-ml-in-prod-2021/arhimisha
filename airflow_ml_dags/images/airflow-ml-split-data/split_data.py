import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split_data")
@click.argument("data_dir")
def split_data(data_dir: str):
    X = pd.read_csv(os.path.join(data_dir, "data.csv"))
    y = pd.read_csv(os.path.join(data_dir, "target.csv"))

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train.to_csv(os.path.join(data_dir, "data_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "data_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "target_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "target_test.csv"), index=False)


if __name__ == '__main__':
    split_data()
