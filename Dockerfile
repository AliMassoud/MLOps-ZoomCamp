FROM jupyter/datascience-notebook

RUN pip freeze
# RUN mkdir -p /MLOps_Camp
# WORKDIR /MLOps_Camp
COPY . .
# RUN apt-get update -y
# RUN apt-get install gcc -y
# RUN pip install -r requirements.txt