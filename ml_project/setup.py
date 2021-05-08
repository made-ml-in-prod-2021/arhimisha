from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Homework 1. ML in production course ',
    author='arhimisha',
    license='MIT',
    install_requires=[
        "click==7.1.2",
        "scikit-learn==0.24.1",
        "dataclasses==0.6",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
    ],
)
