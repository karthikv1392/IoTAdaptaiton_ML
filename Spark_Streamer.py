import os
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.0.2 pyspark-shell'
#    Spark
from pyspark import SparkContext,SparkConf
#    Spark Streaming
from pyspark.streaming import StreamingContext
#    Kafka
from pyspark.streaming.kafka import KafkaUtils
#    json parsing
import json
import time
#import sqlUtils
from elasticsearch import Elasticsearch
from datetime import timedelta
from Initializer import Initialize
import keras
from keras.models import model_from_json
import numpy as np
from numpy import array
import pandas as pd
from datetime import datetime
from sklearn.externals import joblib
from keras import backend as K
import tensorflow as tf
import Spark_Predictor

from Spark_Predictor import Spark_Predictor
#start_time = datetime.now()
prev_vals = [19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0,
                     19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0, 19160.0]   # Initialize the inital energy configuration

#last_read = 0
#sql_connector = sqlUtils.SqlConnector()   # SQL util class

#es = Elasticsearch() # Connect to the default ES
#   init_object = Initialize()
spark_predictor = Spark_Predictor()

model_path = "./model/"
energy_model_file_h5 = "model1.h5"
energy_model_file_json = "model1.json"
traffic_model_file_h5 = "model_traffic_ada.h5"
traffic_model_file_json = "model_traffic_ada.json"

#### scalars #####
scalar_energy = joblib.load("scaler_co.save")
scalar_traffic = joblib.load("scaler_traffic_co.save")

global loaded_model_energy
global loaded_model_traffic
global graph
graph = tf.get_default_graph()
main_list = []
main_list_traffic = []


json_file_energy = open(model_path + energy_model_file_json, 'r')
loaded_model_energy_json = json_file_energy.read()
json_file_energy.close()
loaded_model_energy = model_from_json(loaded_model_energy_json)
# load weights into new model
loaded_model_energy.load_weights(model_path +energy_model_file_h5)
print("Loaded model from disk")

json_file_traffic = open(model_path + traffic_model_file_json, 'r')
loaded_model_traffic_json = json_file_traffic.read()
json_file_traffic.close()
loaded_model_traffic= model_from_json(loaded_model_traffic_json)
# load weights into new model
loaded_model_traffic.load_weights(model_path +traffic_model_file_h5)
print("Loaded model from disk")
#K.clear_session()



#main_df = pd.DataFrame(columns=["timestamp","S11","S13","S5","S7","S4","S12","S8","S2","S10","S3","S6","S1","S9"])

def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    #print(inverted)
    return inverted

def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        #print (forecast)
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        #print (inv_scale)
        # invert differencing
        #print ("length " ,len(series))
        index = len(series) - n_test + i - 1
        #print (index)
        last_ob = series.values[index]

        #print (last_ob)
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted


def ingest(rdd):
    index = 0
    global  start_time
    global prev_vals
    global main_df
    global last_read
    global main_list
    global main_list_traffic
    # for taking the stream of data and processing it
    rows = rdd.map(lambda line: line.strip().split(";"))
    row_list = rows.collect()
    for row in rows.take(rows.count()):
        #print(row[0])
        #print (row[1])
        #data_frame_data = []
        #print (len(row))
        if (len(row)>3):
            time_string = row[0]
            second_level_data = []
            row.pop() # remove the unwanted last element
            vals =  [x1 - float(x2) for (x1, x2) in zip(prev_vals, row[1:])]
            #print (len (vals))
            if (len(vals)==22):
                # Check if we have 22 elements always
                #spark_predictor.main_energy_list.append(vals)
                main_list.append(vals)
                prev_vals = [float(i) for i in row[1:]]

        elif (len(row)==2):
            # This is the case for data traffic
            #spark_predictor.main_data_traffic_list.append(row[1])
            main_list_traffic.append(float(row[1]))

        energy_forecast_total = 0
        data_traffic_forecast = 0
        flag = 0
        if (len(main_list)==10):
            #print (main_list)
            predict_array = np.array(main_list)
            #print (predict_array.shape)
            predict_array = predict_array.reshape(1,10,22)
            with graph.as_default():
                energy_forecast = loaded_model_energy.predict(predict_array)
            #K.clear_session()
            inverse_forecast = energy_forecast.reshape(10, 22)
            inverse_forecast = scalar_energy.inverse_transform(inverse_forecast)
            # print (inverse_forecast)

            inverse_forecast_features = inverse_forecast.reshape(energy_forecast.shape[0], 220)
            for j in range(0, inverse_forecast_features.shape[1]):
                if j not in [0, 22, 44, 66, 88, 110, 132, 154, 176, 198, 6, 28, 50, 72, 94, 116, 138, 160, 182, 204, 9,
                             31, 53, 75, 97, 119, 141, 163, 185, 207, 11, 33, 55, 77, 99, 121, 143, 165, 185, 209,
                             14, 36, 58, 80, 102, 124, 146, 168, 188, 212, 15, 37, 59, 81, 103, 125, 147, 169, 189, 213,
                             18, 40, 62, 84, 106, 128, 150, 172, 192, 216]:
                    energy_forecast_total = energy_forecast_total + inverse_forecast_features[0, j]

            #print (energy_forecast_total)
            flag =1
            #print(main_list)
            #main_list = []
            main_list.pop(0)

        if (len(main_list_traffic)==10):
            predict_array_traffic = np.array(main_list_traffic)
            predict_array_traffic = predict_array_traffic.reshape(1,10,1)
            with graph.as_default():
                traffic_forecasts = loaded_model_traffic.predict(predict_array_traffic)
            #K.clear_session()
            scalar_traffic = joblib.load("scaler_traffic_co.save")
            times = [1,2,3,4,5,6,7,8,9,10]
            current_df_traffic=pd.DataFrame(main_list_traffic,index=times)
            traffic_forecasts = inverse_transform(current_df_traffic, traffic_forecasts, scalar_traffic, n_test=10)
            predicted_traffic = [sum(traffic_forecast) for traffic_forecast in traffic_forecasts]

            data_traffic_forecast = predicted_traffic[0][0]
            #print ("data forecast")
            #print (data_traffic_forecast)

            # remove first element
            main_list_traffic.pop(0)
            flag =1


        if (flag==1):
            print("calling reinforcer")
            print ("len " + str(len(main_list_traffic)))
            print (data_traffic_forecast)
            spark_predictor.predictor(energy_forecast_total,data_traffic_forecast)
            flag=0


    '''
    aggregate_df= pd.read_csv("aggregated_spark.csv",sep=",",index_col="timestamp")  # Read the proccessed data frame
    current_df_co = aggregate_df.iloc[0:10]
    current_df_values_co = current_df_co.values
    test_Set_co = current_df_values_co.reshape((1, 10, 13))
    energy_value_co = spark_predictor.predictor(test_Set_co)
    print (energy_value_co)
    # Drop all rows of main_df
    main_df = pd.DataFrame(columns=["timestamp", "S11", "S13", "S5", "S7", "S4", "S12", "S8", "S2", "S10", "S3", "S6", "S1", "S9"])
    '''


    #print "here"

os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars spark-streaming-kafka-assembly_2.10-1.6.0.jar pyspark-shell'

#batchIntervalSeconds = 10

sc = SparkContext(appName="PythonSparkStreamingKafka_RM_01")
sc.setLogLevel("WARN")
ssc = StreamingContext(sc, 30)


kafkaStream = KafkaUtils.createStream(ssc, 'localhost:2181', 'spark-streaming', {'sensor':1})
#lines = kafkaStream.map(lambda line: line.split(" "))

lines = kafkaStream.map(lambda x: x[1])
line = lines.foreachRDD(lambda rdd : ingest(rdd))




ssc.start()
ssc.awaitTermination()


''''
def creatingFunc():
    ssc = StreamingContext(sc, batchIntervalSeconds)
    # Set each DStreams in this context to remember RDDs it generated in the last given duration.
    # DStreams remember RDDs only for a limited duration of time and releases them for garbage
    # collection. This method allows the developer to specify how long to remember the RDDs (
    # if the developer wishes to query old data outside the DStream computation).
    ssc.remember(60)

if __name__ == '__main__':
    creatingFunc()

'''
