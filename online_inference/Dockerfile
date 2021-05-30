FROM python:3.6

COPY models/model.pkl /model.pkl
COPY models/transformer.pkl /transformer.pkl
COPY models/features_info.yaml /features_info.yaml
COPY setup.py /setup.py
COPY src /src
COPY app_predict.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV PATH_TO_TRANSFORMER="/transformer.pkl"
ENV PATH_TO_FEATURE_INFO="/features_info.yaml"

RUN ["pip", "install", "-e", "."]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]