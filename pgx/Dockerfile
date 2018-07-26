FROM rockdoctor/baseml

RUN apt-get install -y vim
RUN pip3 install confluent-kafka psycopg2 xgboost

COPY src src
COPY models models
