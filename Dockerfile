FROM jupyter/datascience-notebook

USER root


RUN apt-get update && \
    apt-get install -y libpq-dev && \
    apt-get clean && sudo rm -rf var/lib/apt/lists/*

COPY requirements.txt requirements.txt

USER $NB_UID

RUN pip3 install -r requirements.txt && \
    pip3 freeze > libs.txt

