import joblib
import pandas as pd
from kafka import KafkaConsumer
from json import loads
from sqlalchemy import text
from db.engine import engine
from core.config import settings

model = joblib.load('final_model.pkl')

def kafka_consumer():
    consumer = KafkaConsumer(
        'viewer_kafka',
        bootstrap_servers=['kafka:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='viewer_group',
        value_deserializer=lambda x: loads(x.decode('utf-8'))
    )
    return consumer

import time

def consume_and_predict():
    consumer = kafka_consumer()
    print("Consumer created.")
    print(consumer.topics())
    print(consumer.subscription())

    while not consumer.assignment():
        time.sleep(1)
        consumer.poll(timeout_ms=1000)
        print("Waiting for assignment...")

    assignment = consumer.assignment()
    print("Assigned partitions:", assignment)
    if assignment:
        print(consumer.beginning_offsets(assignment))
        print(consumer.end_offsets(assignment))
    print("Consumer started, waiting for messages...")
    for message in consumer:
        print("Message received:", message)
        data = message.value
        if data == '-/Start/-' or data == '-/End/-':
            print("Start/End marker found, continuing...")
            continue
        input_features = data['X']
        data['cols'].remove('y')
        # Convert the dictionary to a list of values
        input_features_list = [input_features[col] for col in data['cols']]
        print("Input features:", input_features_list)
        input_features = pd.DataFrame([data['X']])
        prediction = model.predict(input_features)
        data_to_insert = {col: data['X'][col] for col in data['cols']}
        data_to_insert['y'] = data['y']
        data_to_insert['prediction'] = prediction[0]
        insert_to_db(data_to_insert, (data['cols'] + ['y', 'prediction']))
        print(f"Processed record with prediction: {prediction[0]}")


def insert_to_db(data, cols):
    columns = ', '.join([f'"{col}"' for col in cols])
    placeholders = ', '.join([f':{col}' for col in cols])
    query = text(f"INSERT INTO predictions
                 
                  ({columns}) VALUES ({placeholders})")
    try:
        with engine.connect() as connection:
            result = connection.execute(query, data)
            connection.commit()
            print(f"Inserted {result.rowcount} row(s) into the database successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    consume_and_predict()
