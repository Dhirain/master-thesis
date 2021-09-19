FROM python:slim-buster

COPY odi.csv .

COPY prediction.py .

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python","prediction.py"]