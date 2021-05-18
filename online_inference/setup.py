from setuptools import find_packages, setup

setup(
    name="online_inference",
    packages=find_packages(),
    version="0.1.0",
    description="Homework 2. ML in production course ",
    author="arhimisha",
    install_requires=[
        "scikit-learn==0.24.1",
        "pandas==1.1.5",
        "numpy==1.19.5",
        "fastapi==0.63.0",
        "uvicorn==0.13.4",
        "requests==2.25.1",
        "pytest==6.2.4",
    ],
    license="MIT",
)
