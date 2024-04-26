# etl_workshop_3

                +---------------------------------------+           
                |               Networks                |           
                |                  dev                  |           
                |            driver: bridge             |           
                +---------------------------------------+           
                        |                      |                    
                +-------------+         +-------------+             
                |    Kafka    |         |  Postgres   |             
                |-------------|         |-------------|             
                | Image:      |         | Image:      |             
                | apache/     |         | postgres:13 |             
                | kafka:3.7.0 |         +-------------+             
                | Process:    |               |                     
                | broker,     |         +-------------+             
                | controller  |         |   App-dev   |             
                | Node ID: 1  |         |-------------|             
                | Listeners:  |         | *app-base   |             
                | PLAINTEXT,  |         | Env:        |             
                | CONTROLLER  |         | Gunicorn    |             
                | Advertised: |         | workers,    |             
                | PLAINTEXT   |         | Postgres    |             
                | Volumes:    |         | host        |             
                | kafka-data  |         | Volumes:    |             
                | Ports:      |         | ./app       |             
                | 9092, 9093  |         | Networks:   |             
                +-------------+         | dev         |             
                        |                 +-------------+           
                        |                       |                   
                +-------------+         +-------------+             
                |   Jupyter   |         |   Volume    |             
                |-------------|         |-------------|             
                | Build:      |         | postgres-   |             
                | ./app       |         | db-volume   |             
                | Command:    |         | kafka-      |             
                | Jupyter     |         | data        |             
                | notebook    |         +-------------+             
                | Ports:      |                                     
                | 8888        |                                     
                | Networks:   |                                     
                | dev         |                                     
                | Depends:    |                                     
                | app-dev     |                                     
                +-------------+                                     

- **Kafka**: Running as both broker and controller, with listeners for PLAINTEXT and CONTROLLER communications. It uses volume `kafka-data` for storage.
- **Postgres**: Using PostgreSQL image version 13, linked to the `postgres-db-volume` for data persistence.
- **App-dev**: Inherits base app configurations, linked to Postgres and Kafka, and mounts `./app` directory.
- **Jupyter**: Runs a Jupyter Notebook server, dependent on `app-dev`, also mounted on `./app`.

## Up & running!
Rename the ***_template_env_local*** to ***.env*** and update with your values.

In the root folder execute the next in order to initialize wour workspace.

### Start UI

docker compose up -d

## Cleaning-up the environment

docker compose down --volumes --remove-orphans

## Go on!


## References
[Baeldung Kafka Docker Setup](https://www.baeldung.com/ops/kafka-docker-setup)

[Machine Learning Mastery XGBoost for Regression](https://machinelearningmastery.com/xgboost-for-regression/)