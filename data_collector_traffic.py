_Author_ = "*******"

# Data traffic path

from Initializer import Initialize
from datetime import datetime
from datetime import timedelta
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
init_object = Initialize()


class DataLoader():
    # class to load the csv data to json
    # This will also support loading real-time simulation data from kafka queue for prediction

    data_traffic_path = ""
    data_traffic_file = ""
    def __init__(self):
        self.data_traffic_path = init_object.data_traffic_path
        self.data_traffic_file = init_object.data_traffic_file


    def load_data_to_csv(self):
        # loads the log file to csv file
        file =  open(self.data_traffic_path + self.data_traffic_file,"r")
        time_data_frame  = {}
        traffic_count  = 0
        traffic_induvidual_count = 0 # For each time instance
        prev_time   = 0.0 # Keep a check on the time
        df_dict = {}
        line_count = 0
        for line in file.readlines():
            if "Time" in line:
                if line_count>0:
                    df_dict[prev_time]=traffic_induvidual_count
                current_time = float(line.split(":")[1].split(" ")[1])

                #print (current_time)
                prev_time = current_time
                traffic_induvidual_count = 0

                line_count += 1
            if ("finished sending") in line:
                # Count the traffic
                #if ("S11") or ("S47") or("S48") or ("") not in line:
                # The database
                traffic_induvidual_count += 1
                #print (line)
                traffic_count += 1

            if ("receiving") in line:
                #if ("S11") not in line:
                # Count the traffic
                traffic_induvidual_count += 1
                # print (line)
                traffic_count += 1

        df_dict[prev_time]=traffic_induvidual_count
        max_time =prev_time  # The last time value inserted becomes the maximum time
        start_time = datetime.now() - timedelta(seconds=max_time)

        new_df_dict = {}
        check_Sum = 0
        dataframe_dict = {}
        dataframe_dict["timestamp"] = []
        dataframe_dict["traffic"] = []
        for key in df_dict.keys():
            check_Sum =  check_Sum + df_dict[key]
            #print (df_dict[key])
            milliseconds = float(key *1000)
            #print (milliseconds)
            time_value = start_time + timedelta(milliseconds=milliseconds)
            dataframe_dict["timestamp"].append(time_value)
            #print(time_value)
            #print (time_value)
            timestamp = int(time_value.timestamp()*1000)
            if time_value in new_df_dict:
                new_df_dict[time_value] = new_df_dict[time_value] +  df_dict[key]
                dataframe_dict["traffic"].append(new_df_dict[time_value] +  df_dict[key])

            else:
                new_df_dict[time_value] = df_dict[key]
                dataframe_dict["traffic"].append(df_dict[key])

        #print (check_Sum)
        #print (traffic_count)

        #print(len(df_dict.keys()))

        #print (len(new_df_dict.keys()))
        #print (sum(new_df_dict.values()))


        processed_dataframe = pd.DataFrame(dataframe_dict)
        #print (processed_dataframe)
        processed_dataframe.index = processed_dataframe["timestamp"]
        # aggregate the dataframe now for one minute intervals
        aggregate_df = processed_dataframe.resample('1T').sum()
        #aggregate_df.to_csv(init_object.data_path+ "aggregated_traffic_" + self.data_traffic_file.split("_")[1] + "_su"+ ".csv",index =True)
        aggregate_df.to_csv(init_object.data_path+ "aggregated_traffic_" +".csv",index =True)

        #print (aggregate_df)
        #print (init_object.data_path+ "aggregated_traffic_" + self.data_traffic_file.split("_")[1] + ".csv")
        print ("Data Generation Complete")
        plt.plot(aggregate_df["traffic"])
        plt.savefig("trafficplot.png")



if __name__ == '__main__':

    data_loader_object = DataLoader()
    data_loader_object.load_data_to_csv()

