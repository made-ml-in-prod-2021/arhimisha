import os
import click
import pandas as pd


@click.command("check_data")
@click.argument("data_dir")
def check_data(data_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    target = pd.read_csv(os.path.join(data_dir, "target.csv"))

    if not data.shape[1] == 4:
        raise Exception(f"Wrong number of features! Expected is 4. Received is {data.shape[1]}.")
    if not target.shape[1] == 1:
        raise Exception(f"Wrong number of target! Expected is 1. Received is {target.shape[1]}.")


if __name__ == '__main__':
    check_data()
