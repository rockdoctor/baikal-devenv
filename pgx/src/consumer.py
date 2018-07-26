from confluent_kafka import Consumer, KafkaError, Producer
import psycopg2
import json
import pandas as pd
import warnings
import pickle
import xgboost as xgb
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Kafka Settings
settings = {
    'bootstrap.servers': '10.6.0.155:9092',
    'group.id': 'mygroup',
    'client.id': 'client-1',
    'enable.auto.commit': True,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'latest'}
}

# DB config
conn_string = "host='postgres' dbname='admin' user='admin' password='admin'"
conn = psycopg2.connect(conn_string)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS queue (id serial PRIMARY KEY, num integer, data varchar);")
conn.commit()

c = Consumer(settings)

c.subscribe(['pgx'])
p = Producer({'bootstrap.servers': '10.6.0.155:9092'})

with open('/models/PGXModel.dat', 'rb') as f:
    model = pickle.load(f)
with open('/models/PGXScaler.dat', 'rb') as f:
    scaler = pickle.load(f)
columns = model.feature_names
try:
    while True:
        msg = c.poll(0.1)
        if msg is None:
            continue
        elif not msg.error():
            data = json.loads(msg.value().decode('utf-8'))
            df = pd.read_json('['+json.dumps(data)+']', orient='records')
            patid = df.pop('patid')
            df = df[columns]
            df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
            matrix = xgb.DMatrix(df, feature_names=df.columns)
            probs = model.predict(matrix)
            print('Patid: {0}   Prediction: {1}'.format(patid[0], probs))
            df['patid'] = patid
            df['warfarin'] = probs[0]
            df = df.to_dict('records')
            p.produce('pgx-result', value=json.dumps(df[0]))
            p.flush()
            c.commit()
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            print('End of partition reached {0}/{1}'
                  .format(msg.topic(), msg.partition()))
        else:
            print('Error occured: {0}'.format(msg.error().str()))

except KeyboardInterrupt:
    pass

finally:
    c.close()
