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
import hydrostats as hs
import scipy
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

def error_stats(forecast_datapath,forecast_number,output_datapath):
    #format dataframe for metrics
    df = pd.read_csv(forecast_datapath)
    df['recorded streamflow'] = df['recorded streamflow'].astype(np.float64)

    # Replace 0's with .001
    df = df.replace(to_replace=0, value=0.001)
    df = df.replace(to_replace=1, value=1.0001)

    # Log Transform Data
    df['log Predicted'] = np.log(df['predicted streamflow'])
    df['log Recorded'] = np.log(df['recorded streamflow'], dtype='float64')

    # Account for any negatives:
    if min(df['log Predicted']) < 0 or min(df['log Recorded']) < 0:
        adjustment = max(abs(min(df['log Predicted'])), abs(min(df['log Recorded']))) + 1
        print(adjustment)
        df['log Predicted'] = df['log Predicted'] + adjustment
        df['log Recorded'] = df['log Recorded'] + adjustment
    else:
        adjustment = 0

    ## create Dataframe columns
    cor_coeff = {}
    mean_diff = {}
    mean_var = {}
    month_predicted = {}
    month_observed = {}
    rmse = {}
    rmse_log = {}
    NS_eff = {}
    R2 = {}
    sa = {}

    #Total Metrics
    log_pred = df.dropna()['log Predicted']
    log_observed = df.dropna()['log Recorded']
    cor_coeff[1] = hs.acc(log_observed, log_pred)
    mean_diff[1] = np.exp(scipy.stats.gmean(log_observed) - scipy.stats.gmean(log_pred))
    mean_var[1] = np.var(log_observed - log_pred)
    month_predicted[1] = np.exp(np.mean(log_pred - adjustment))
    month_observed[1] = np.exp(np.mean(log_observed - adjustment))

    #Error Metrics
    predicted = df.dropna()['predicted streamflow']
    observed = df.dropna()['recorded streamflow']
    log_predicted = df.dropna()['log Predicted']
    log_observed = df.dropna()['log Recorded']
    n = predicted.count()
    rmse[1] = hs.rmse(predicted, observed)
    rmse_log[1] = hs.rmsle(log_predicted, log_observed)
    NS_eff[1] = hs.E(log_predicted, log_observed)
    R2[1] = hs.r_squared(predicted, observed)
    sa[1] = hs.sa(predicted, observed)

    #Combine into one dataframe
    cor_coeff_df = pd.DataFrame.from_dict(cor_coeff, orient='Index')
    mean_df = pd.DataFrame.from_dict(mean_diff, orient='Index')
    variance_df = pd.DataFrame.from_dict(mean_var, orient='Index')
    predicted_df = pd.DataFrame.from_dict(month_predicted, orient='Index')
    observed_df = pd.DataFrame.from_dict(month_observed, orient='Index')
    correlation_df = pd.concat([cor_coeff_df, mean_df, observed_df, predicted_df, variance_df], axis=1)
    correlation_df.columns = ['Correlation', 'Mean Difference', 'Observed Flow', 'Predicted Flow',
                              'Mean Variance']
    rmse_df = pd.DataFrame.from_dict(rmse, orient='Index')
    rmse_log_df = pd.DataFrame.from_dict(rmse_log, orient='Index')
    NS_eff_df = pd.DataFrame.from_dict(NS_eff, orient='Index')
    R2_df = pd.DataFrame.from_dict(R2, orient='Index')
    sa_df = pd.DataFrame.from_dict(sa, orient='Index')
    error_df = pd.concat([rmse_df, rmse_log_df, NS_eff_df, R2_df, sa_df], axis=1)
    error_df.columns = ['RMSE', 'Log RMSE', 'Nash-Sutcliffe Efficiency', 'R^2 Coefficient', 'Spectral Angle']
    results_df = pd.concat([correlation_df, error_df], axis=1)

    results_df['Forecast']="Forecast "+str(forecast_number)
    print(results_df)

    #print results to .csv

    datapath_results=output_datapath+"Forecast_"+str(forecast_number)+".csv"
    print(datapath_results)
    results_df.to_csv(datapath_results,sep=',', index=False)
    return results_df

def combine_csvs(folderpath):
    results=glob.glob(folderpath+"Forecast_*")
    results.sort(key=sort_key)
    df_list=[]
    for file in results:
        df_list.append(pd.read_csv(file))
    df_results=pd.concat(df_list)
    df_results.to_csv(folderpath+"Statistical Summary.csv", sep=',', index=False)

counter=1
for i in spt_forecast:
    forecast=folderpath + str(i)
    print(forecast)
    data_format(forecast,counter)
    counter+=1
results_path=basepath+'\\Results\\Merged Files\\'
combine_forecasts(results_path)

forecasts=[os.path.basename(x) for x in glob.glob(results_path + "*_merged.csv")]
forecasts.sort(key=sort_key)

stats_summary_path=basepath+'\\Results\\Summary Files\\'
if not os.path.exists(basepath+'\\Results\\Summary Files\\'):
    os.makedirs(basepath+'\\Results\\Summary Files\\')

forecast_number=1

for i in forecasts:
    input_datapath=results_path+str(i)
    error_stats(input_datapath,forecast_number,stats_summary_path)
    forecast_number+=1
    print(forecast_number)

combine_csvs(stats_summary_path)
