_Author_ = "***********"

import subprocess

import csv
import sys
import time
from kafka import KafkaConsumer, KafkaProducer

class kafka_producer():
    def publish_message(self,producer_instance, topic_name, key, value):
        try:
            key_bytes = bytearray(key,'utf8')
            value_bytes = bytearray(value,'utf8')
            producer_instance.send(topic_name, key=key_bytes, value=value_bytes)
            producer_instance.flush()
            print('Message published successfully.')
        except Exception as ex:
            print('Exception in publishing message')
            print(str(ex))

    def connect_kafka_producer(self):
        _producer = None
        try:
            _producer = KafkaProducer(bootstrap_servers=['localhost:9092'], api_version=(0, 10))
        except Exception as ex:
            print('Exception while connecting Kafka')
            print(str(ex))
        finally:
            return _producer

producer_object = kafka_producer()




def stream_csv_file():
    # read and stream csv files
    producer_instance = producer_object.connect_kafka_producer()
    with open('$CUPPROJECTPATH/results/wisen_simulation.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        while (True):
            for row in csv.reader(iter(csv_file.readline,'')):
                if len(row)>0:
                    line_data = row[0].strip("\n")
                    #print (line_data)
                    #print (line_data.split(";"))
                    if(len(line_data.split(";"))>23):
                        # Sent the QoS data obtained from CupCarbon
                        #print (row[0])
                        if not "Time" in line_data:
                            print (len(line_data.split(";")))
                            #time.sleep(1)
                            producer_object.publish_message(producer_instance,"sensor","data",line_data)



                #time.sleep(1)
if __name__ == '__main__':
    stream_csv_file()
