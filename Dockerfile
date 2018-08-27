FROM python:2.7

RUN apt-get update && apt-get install -y npm netcat nodejs-legacy

RUN install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev

RUN pip install --upgrade pip  # make things faster, hopefully

COPY codalab/requirements/common.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app/codalab
