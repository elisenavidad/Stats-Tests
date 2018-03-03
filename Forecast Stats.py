"""
Statistical analysis of ensemble forecast netCDF files
By: Elise Jackson
"""

import netCDF4 as nc
from netCDF4 import *
import glob
import pandas as pd
import numpy as np
import os
from merge_data_script import merge_data
import re
import csv
import datetime as dt


def sort_key(str):
    return int("".join(re.findall("\d*", str)))

#set file paths
basepath=os.path.dirname(os.path.abspath(__file__))
foldername="\\20170608.12\\"
folderpath=os.path.dirname(os.path.abspath(__file__))+foldername
print (folderpath)
spt_forecast=[os.path.basename(x) for x in glob.glob(folderpath+"*.nc")]
spt_forecast.sort(key=sort_key)

#file for testing
file="D:\\Jackson\\Forecast Stats\\Python\\20170608.12\\Qout_brazil_itajai_acu_historical_3.nc"

#Set River ID
stream_ID=382

def data_format(file,counter):
    print("Counter = " +str(counter))

    #Extract flow and times from NetCDF file
    ncf=nc.Dataset(file)
    rivid=ncf.variables['rivid'][:]
    riv_index=rivid.tolist().index(stream_ID)
    print(riv_index)
    flow=ncf.variables['Qout'][:,riv_index]
    times=ncf.variables['time'][:]
    length=min(len(flow),len(times))
    print(length)
    flow=flow[:length]
    times=times[:length]

    #Generate Pandas dataframe from flow and times
    d={'Time':times, 'Discharge':flow}
    df=pd.DataFrame(data=d)
    df['Time']=pd.to_datetime(df['Time'],unit='s')
    cols=df.columns.tolist()
    cols=cols[-1:]+cols[:-1]
    df=df[cols]
    print(df)

    #Print extracted flow to temporary csv
    datapath_results=basepath+"//Results//"+"tempflow.csv"
    print (datapath_results)
    df.to_csv(datapath_results,sep=',',index=False)

    #Merge csv with observed data
    recorded_forecast=basepath+"\\Observed Data\\"+'Observed_DataBlumenau.csv'
    predicted_forecast=datapath_results
    location=basepath+'\\Results\\'+'\\Merged Files\\'+"Blumenau_"+str(stream_ID)+"_"+str(counter)
    if not os.path.exists(basepath+'\\Results\\'+'\\Merged Files\\'):
        os.makedirs(basepath+'\\Results\\'+'\\Merged Files\\')
    merge_data(recorded_forecast,predicted_forecast,location)
    merged_file=location+"_merged.csv"

    #delete temporary flow file of extracted data
    os.remove(datapath_results)

    #Create new dataframe from .csv file
    df_compare=pd.read_csv(merged_file)
    return df_compare

def ensemble_plot(dataframe, basepath):
    plt_filename=basepath+"\\Plots\\"+"Summary Plot.png"
    if not os.path.exists(basepath+"\\Plots"):
        os.makedirs(basepath+"\\Plots\\")
    return

def combine_forecasts(folderpath):
    counter=1
    #Grab all the forecast files
    forecasts=glob.glob(folderpath+"*_merged.csv")
    forecasts.sort(key=sort_key)
    #Start new dataframe from first forecast
    frame=pd.read_csv(forecasts[0])
    #Format frame to make it look pretty
    frame=frame[['Datetime','recorded streamflow','predicted streamflow']]
    frame.columns=['Datetime','Recorded Streamflow', 'Forecast '+str(counter)]
    frame=frame.set_index('Datetime')
    print(frame)
    counter+=1
    #Loop through other files and add them to make a super dataframe of all the forecasts
    for file in forecasts[1:51]:
        df=pd.read_csv(file,index_col=0)
        # print(df)
        frame=pd.concat([frame,df['predicted streamflow']], axis=1)
        #Rename new predicted streamflow column to "Forecast #"
        frame=frame.rename(columns={'predicted streamflow':'Forecast '+str(counter)})
        print(frame)

        counter+=1

        basefile=basepath+'\\Results\\'+"Forecast Summary.csv"
        print(basefile)
        frame.to_csv(basefile,sep=',',index=True)

    return frame


def ensemble_stats(file):
    return file

def error_stats(df):
    return df


# counter=1
# for i in spt_forecast:
#     forecast=folderpath + str(i)
#     print(forecast)
#     data_format(forecast,counter)
#     counter+=1
results_path=basepath+'\\Results\\Merged Files\\'
combine_forecasts(results_path)

