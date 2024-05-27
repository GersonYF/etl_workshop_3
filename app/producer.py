import time
import pandas as pd
from kafka import KafkaProducer
from json import dumps
from db.engine import engine
from core.config import settings
from sklearn.model_selection import train_test_split

def kafka_producer():
    producer = KafkaProducer(
        value_serializer=lambda x: dumps(x).encode('utf-8'),
        bootstrap_servers=['kafka:9092']
    )
    return producer

def on_send_success(record_metadata):
    print(f'Message sent to topic {record_metadata.topic} partition {record_metadata.partition} at offset {record_metadata.offset}')

def on_send_error(excp):
    print('I am an errback', exc_info=True)

def data_streaming():
    topic = 'viewer_kafka'
    dataframe_to_stream = pd.read_sql_table(settings.CLEAN_TABLE, con=engine)

    X = dataframe_to_stream.drop('y', axis=1)
    y = dataframe_to_stream['y']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    producer = kafka_producer()
    producer.send(topic, value='-/Start/-').add_callback(on_send_success).add_errback(on_send_error)

    for index in range(len(X_test)):
        x_row = X_test.iloc[index].to_dict()
        y_value = y_test.iloc[index]
        producer.send(
            topic,
            value={
                'X': x_row,
                'y': y_value,
                'cols': dataframe_to_stream.columns.tolist()
            }).add_callback(on_send_success).add_errback(on_send_error)
        time.sleep(1)

    producer.send(topic, value='-/End/-').add_callback(on_send_success).add_errback(on_send_error)
    producer.flush()
    producer.close()
    print('End streaming.')

if __name__ == "__main__":
    data_streaming()
