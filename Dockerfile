FROM python:2.7

# For nodejs
RUN apt-get update
RUN apt-get install -y nodejs-legacy build-essential
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash -

#RUN curl -sL https://deb.nodesource.com/setup_4.x | bash -
RUN apt-get install -y npm netcat
RUN apt-get update && apt-get install -y nodejs python-dev libmemcached-dev

RUN pip install --upgrade pip  # make things faster, hopefully

COPY codalab/requirements/common.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

WORKDIR /app/codalab
