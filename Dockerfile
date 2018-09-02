FROM python:2.7

RUN apt-get update
RUN apt-get install -y nodejs-legacy build-essential
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -
RUN apt-get install -y npm netcat

RUN apt-get install -y gfortran libatlas-base-dev liblapacke-dev python3-dev

RUN pip install --upgrade pip  # make things faster, hopefully

COPY codalab/requirements/libact.txt libact_req.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r libact_req.txt
COPY codalab/requirements/common.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

WORKDIR /app/codalab
