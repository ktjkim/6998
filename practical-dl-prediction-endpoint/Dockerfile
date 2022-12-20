FROM python:3.7

USER root

ADD . /home/model-server/

WORKDIR /home/model-server/

RUN pip3 install --upgrade pip

RUN pip install -r requirements.txt

CMD exec gunicorn -b :5000 --max-requests 1 --graceful-timeout 300 -t 600 main:app