import shutil
import click


@click.command("prepare_data")
@click.argument("input_dir")
@click.argument("output_dir")
def prepare_data(input_dir: str, output_dir: str):
    shutil.copytree(input_dir, output_dir)


if __name__ == '__main__':
    prepare_data()
