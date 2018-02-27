import csv
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from os import listdir

def merge_data(recorded_data,interim_data,location):
    #Importing data into a dataframe
    df_recorded = pd.read_csv(recorded_data, delimiter=",", header=None, names=['recorded streamflow'], index_col=0, infer_datetime_format=True, skiprows=1)
    df_predicted = pd.read_csv(interim_data, delimiter=",", header=None, names=['predicted streamflow'], index_col=0, infer_datetime_format=True, skiprows=1)
    #Converting the index to datetime type
    df_recorded.index = pd.to_datetime(df_recorded.index, infer_datetime_format=True)
    df_predicted.index = pd.to_datetime(df_predicted.index, infer_datetime_format=True)
    #Joining the two dataframes
    df_merged = pd.DataFrame.join(df_predicted, df_recorded).dropna()
    df_merged.to_csv(location + "_merged.csv",sep=",",index_label="Datetime")



# recorded_list = listdir('C:\\Users\\wadear\\Documents\\Nepal_Data\\recorded_raw_data')
# interim_list = listdir('C:\\Users\\wadear\\Documents\\Nepal_Data\\interim_raw_data')
# locations = ['Asaraghat','Babai','Bheri','Kaligandaki','Kamali','Kankai','Marsyangdi','Narayani','Rapti','Saptakosi','Seti','Tinaukhola']
#
# print(recorded_list)
# print(interim_list)
# print(locations)
#
# for i,j,k in zip(recorded_list,interim_list,locations):
#     merge_data(i,j,k)


