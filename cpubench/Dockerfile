FROM ubuntu:20.04

RUN apt-get update && apt-get install -y sysbench

CMD ["sysbench","cpu","--threads=4","run"]