FROM python:slim-buster

COPY odi.csv .

COPY mini_prediction.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","mini_prediction.py"]