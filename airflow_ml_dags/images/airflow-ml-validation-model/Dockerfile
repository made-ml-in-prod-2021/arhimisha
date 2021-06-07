FROM python:3.6

RUN mkdir /build/

COPY requirements.txt /build/requirements.txt

RUN pip install -r /build/requirements.txt

COPY validation_model.py /validation_model.py

ENTRYPOINT ["python", "validation_model.py"]
