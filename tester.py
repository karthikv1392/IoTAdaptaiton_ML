import numpy as np
import pandas as pd
import time

import subprocess
import os
import matplotlib
matplotlib.use('Agg')
from subprocess import STDOUT,PIPE
from keras.models import model_from_json
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv
from numpy import array
from sklearn.externals import joblib
import pysftp
import traceback
def parser(x):

    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

# Load the dataset into the memory and start the learning
def read_data(filepath):
    # Takes as input the path of the csv which contains the descriptions of energy consumed

    aggregated_df = read_csv(filepath, header=0, parse_dates=[0], index_col=0,
                             squeeze=True, date_parser=parser)

    aggregated_series =  aggregated_df.values # Convert the dataframe to a 2D array and pass back to the calling function

    return aggregated_series


def test():
    np.random.seed(1618033)

    #Set 3 axis labels/dims
    years = np.arange(2000,2010) #Years
    samples = np.arange(0,20) #Samples
    patients = np.array(["patient_%d" % i for i in range(0,3)]) #Patients

    #Create random 3D array to simulate data from dims above
    A_3D = np.random.random((years.size, samples.size, len(patients))) #(10, 20, 3)

    # Create the MultiIndex from years, samples and patients.
    midx = pd.MultiIndex.from_product([years, samples, patients])

    # Create sample data for each patient, and add the MultiIndex.
    patient_data = pd.DataFrame(np.random.randn(len(midx), 3), index = midx)

    series = patient_data.values

    #print (series)
    #print (series[:,[-2,-1]])
    series = series[:10,:]

    series_2 = series.reshape(30,1)
    print (series_2)

    count = 0

    while (1):
        f = open("sample.txt","w")
        f.write(str(count))
        count += 1
        time.sleep(2)


def calculate_total_traffic():
    # To calculate the total data traffic of different self-adaptive patterns
    df_co_traffic = pd.read_csv(
        "./data/aggregated_traffic_day_co.csv",
        sep=",",
        index_col="timestamp")

    df_su_traffic = pd.read_csv(
        "./data/aggregated_traffic_day_su.csv",
        sep=",",
        index_col="timestamp")

    df_sc_traffic = pd.read_csv(
        "./data/aggregated_traffic_sc_day.csv",
        sep=",",
        index_col="timestamp")

    df_reinforce_traffic = pd.read_csv(
        "./data/aggregated_traffic_reinforce_day.csv",
        sep=",",
        index_col="timestamp")


    modified_df_co = df_co_traffic.head(1440)
    modified_df_su = df_su_traffic.head(1440)
    modified_df_sc = df_sc_traffic.head(1440)
    modified_df_reinforce = df_reinforce_traffic.head(1440)


    print ("Collect traffic " ,sum(modified_df_co["traffic"]))
    print ("Synth Utilize traffic " ,sum(modified_df_su["traffic"]))
    print ("Synth Command traffic " ,sum(modified_df_sc["traffic"]))
    print ("Reinforce Traffic " ,sum(modified_df_reinforce["traffic"]))


    co = [18959,19002,19014,19017,18987,18857,19022,19025,18947,18932,18349,18995,18995,18965,19008]

    su = [18773,19008,18875,18887,18996,18568,18888,18902,18957,18716,18786,19003,19002,18784,18864]

    sc = [18959,19002,19014,19017,18987,18857,19022,19025,18947,18932,18967,18995,18995,18965,19008]

    sum_val = 0

    for val in co:
        sum_val = sum_val + (19160-val)

    print("Collect Energy ", sum_val)

    sum_val = 0

    for val in su:
        sum_val = sum_val + (19160 - val)

    print("Synth Util ", sum_val)

    sum_val = 0

    for val in sc:
        sum_val = sum_val + (19160 - val)

    print("Synth Command ", sum_val)




def count_adaptations():
    f = open("pattern_spark.txt","r")
    prev = ""
    count =0
    for line in f.readlines():
        line = line.strip("\n")
        if prev!=line:
            count +=1
        prev = line
    print (count)

def count_threshold_exceedences():


    f = open("./data/aggregated_traffic_reinforce.csv")

    # Create four lists for data traffic
    co_list = []
    su_list = []
    sc_list  = []
    reinforce_list = []
    max_limit = 400
    min_limit = 200
    max_count_reinforce = 0
    min_count_reinforce = 0
    line_count  = 0
    for line in f.readlines():

        if (line_count>0 and line_count<=1440):
            traffic = line.split(",")
            if (int(traffic[1]) > max_limit):
                max_count_reinforce+=1
            if (int(traffic[1]) < min_limit):
                min_count_reinforce +=1
            reinforce_list.append(int(traffic[1]))
        line_count +=1

    print ("Reinforce max " , max_count_reinforce)
    print ("Reinforce min", min_count_reinforce)

    f = open("./data/aggregated_traffic_day_co.csv")

    max_count_reinforce = 0
    min_count_reinforce = 0
    line_count = 0
    for line in f.readlines():

        if (line_count > 0 and line_count<=1440):
            traffic = line.split(",")
            if (int(traffic[1]) > max_limit):
                max_count_reinforce += 1
            if (int(traffic[1]) < min_limit):
                min_count_reinforce += 1
            co_list.append(int(traffic[1]))
        line_count += 1

    print("CO max ", max_count_reinforce)
    print("CO min", min_count_reinforce)

    f = open("./data/aggregated_traffic_day_su.csv")

    max_count_reinforce = 0
    min_count_reinforce = 0
    line_count = 0
    for line in f.readlines():

        if (line_count > 0 and line_count <=1440):
            traffic = line.split(",")
            if (int(traffic[1]) > max_limit):
                max_count_reinforce += 1
            if (int(traffic[1]) < min_limit):
                min_count_reinforce += 1
            su_list.append(int(traffic[1]))
        line_count += 1

    print("SU max ", max_count_reinforce)
    print("sU min", min_count_reinforce)

    f = open("./data/aggregated_traffic_day_sc.csv")

    max_count_reinforce = 0
    min_count_reinforce = 0
    line_count = 0
    for line in f.readlines():

        if (line_count > 0 and line_count<=1440):
            traffic = line.split(",")
            if (int(traffic[1]) > max_limit):
                max_count_reinforce += 1
            if (int(traffic[1]) < min_limit):
                min_count_reinforce += 1
            sc_list.append(int(traffic[1]))
        line_count += 1

    print("Sc max ", max_count_reinforce)
    print("sc min", min_count_reinforce)

    pyplot.plot(co_list, label="CO")
    pyplot.plot(su_list, label="SU")
    pyplot.plot(sc_list, label="SC")
    pyplot.plot(reinforce_list, label="ourApproach", color='y')
    pyplot.axhline(y=400.0, color='r', linestyle='-')
    pyplot.axhline(y=200.0, color='r', linestyle='-')
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Time (Minutes)")
    pyplot.ylabel("# message exchanges")
    # pyplot.axis([0, 100, 1, 20])
    pyplot.savefig("plot_traffic_exceedences.png", transparent="True", dpi=300, quality=95)
    # pyplot.axhline(y=0.5, color='r', linestyle='-')


def count_energy_exceedance_thresholds():
    last_read = 0
    df_energy_reinforce= pd.read_csv("./data/aggregated_energy_reinfofce_day.csv",sep=",",
        index_col="timestamp")

    excedence_count = 0
    min_exceedence_count  = 0
    max_value = 14.0
    min_value  = 9.0

    # Create four lists for the graph

    reinforce_list = []
    co_list = []
    su_list  = []
    sc_list  = []

    energy_j = 0
    while(last_read+10<=1440):
        current_df_sc = df_energy_reinforce.iloc[last_read:last_read + 10]
        current_df_sc_values = current_df_sc.values
        for i in range(0, 10):
            for j in range(0, 22):
                # Remove the sensor components
                if j not in [6, 9, 11, 14, 15, 18]:
                    energy_j = energy_j + current_df_sc_values[i, j]

        reinforce_list.append(energy_j)

        if (energy_j > max_value):
            excedence_count +=1
        if (energy_j <=min_value):
            min_exceedence_count += 1

        energy_j = 0
        last_read = last_read + 1

    print ("Reinforce ", excedence_count)
    print ("Reinforce min ", min_exceedence_count)
    df_energy_reinforce = pd.read_csv("./data/aggregated_energy_co_day.csv", sep=",",index_col="timestamp")


    excedence_count = 0
    min_exceedence_count = 0
    last_read = 0
    energy_j = 0
    while (last_read + 10 <= 1440):
        current_df_sc = df_energy_reinforce.iloc[last_read:last_read + 10]
        current_df_sc_values = current_df_sc.values
        for i in range(0, 10):
            for j in range(0, 22):
                if j not in [6, 9, 11, 14, 15, 18]:
                    energy_j = energy_j + current_df_sc_values[i, j]

        co_list.append(energy_j)
        if (energy_j > max_value):
            excedence_count += 1
        if (energy_j <=min_value):
            min_exceedence_count += 1
        energy_j = 0
        last_read = last_read + 1


    print ("co " , excedence_count)
    print ("co min" , min_exceedence_count)

    df_energy_reinforce = pd.read_csv(
        "./data/aggregated_energy_su_day.csv", sep=",",
        index_col="timestamp")

    excedence_count = 0
    min_exceedence_count = 0
    last_read = 0
    energy_j = 0
    while (last_read + 10 <= 1440):
        current_df_sc = df_energy_reinforce.iloc[last_read:last_read + 10]
        current_df_sc_values = current_df_sc.values
        for i in range(0, 10):
            for j in range(0, 22):
                if j not in [6, 9, 11, 14, 15, 18]:
                    energy_j = energy_j + current_df_sc_values[i, j]

        su_list.append(energy_j)
        if (energy_j > max_value):
            excedence_count += 1

        if (energy_j <=min_value):
            min_exceedence_count += 1
        energy_j = 0
        last_read = last_read + 1


    print("su ", excedence_count)
    print("su  min", min_exceedence_count)

    df_energy_reinforce = pd.read_csv(
        "./data/aggregated_energy_sc_day.csv", sep=",",
        index_col="timestamp")

    excedence_count = 0
    min_exceedence_count = 0
    last_read = 0
    energy_j = 0
    while (last_read + 10 <= 1440):
        current_df_sc = df_energy_reinforce.iloc[last_read:last_read + 10]
        current_df_sc_values = current_df_sc.values
        for i in range(0, 10):
            for j in range(0, 22):
                if j not in [6, 9, 11, 14, 15, 18]:
                    energy_j = energy_j + current_df_sc_values[i, j]

        sc_list.append(energy_j)
        if (energy_j > max_value):
            excedence_count += 1
        if (energy_j <=min_value):
            min_exceedence_count += 1
        energy_j = 0
        last_read = last_read + 1

    print("sc ", excedence_count)
    print("sc min ", min_exceedence_count)


    pyplot.plot(co_list, label="CO")
    pyplot.plot(su_list, label="SU")
    pyplot.plot(sc_list,label="SC")
    pyplot.plot(reinforce_list,label="ourApproach",color='y')
    pyplot.axhline(y=14.0, color='r', linestyle='-')
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Time (Minutes)")
    pyplot.ylabel("Energy Consumption (Joules)")
    # pyplot.axis([0, 100, 1, 20])
    pyplot.savefig("plot_energy_exceedences.png", transparent="True", dpi=300, quality=95)
    #pyplot.axhline(y=0.5, color='r', linestyle='-')



def copy_network():
    # To copy files to a remote work
    srv = pysftp.Connection(host="192.168.1.9", username="**********",
                            password="incorrectpass")


    remote_file ="adaptation.txt"

    f=open("adaptation.txt","r")


    srv.put(remote_file)

    srv.close()


if __name__ == '__main__':
    #test_model()
    #test_model_traffic()
    #calculate_total_traffic()
    #make_indexex_one_traffic()
    #copy_network()
    count_adaptations()
    count_threshold_exceedences()
    count_energy_exceedance_thresholds()


