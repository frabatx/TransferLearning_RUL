FROM tensorflow/tensorflow:latest-gpu-jupyter

USER root


RUN apt-get update --fix-missing
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

USER $NB_UID

RUN pip3 install -r requirements.txt && \
    pip3 freeze > libs.txt

