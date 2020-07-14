FROM python:3.6

RUN apt-get update && \
    apt-get install -y build-essential cmake libboost-all-dev python3-dev

ADD ./requirements.txt /srv/requirements.txt
RUN pip3 install -r /srv/requirements.txt
ADD . /srv

WORKDIR /srv

EXPOSE 8000

CMD ["python", "face_recognition_api.py"]
