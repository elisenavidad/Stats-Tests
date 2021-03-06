"""
Paired T-test Statistical Analysis
Generates the Correlation Coefficient and p-value for a paired t-test of predicted streamflow and observed streamflow
Calculates the geometric mean (Mean Difference) between predicted and observed streamflow.
    This is a multiplicative factor between predicted and observed.  Predicted*mean difference = Observed Flow
Calculates the variance of the mean difference
Creates individual csv files for each station analyzed, as well as a 'National Results' summary file.
Creates plots of predicted vs. observed streamflow for each station analyzed.
Written by: Elise Jackson
3-8-18
"""
import csv
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import hydrostats as hs



# Set variable paths
# stations = ['Asaraghat']     #for testing one station
# specify folder where merged data files are, in the same directory as this script
foldername="\\Test\\"
folderpath=os.path.dirname(os.path.abspath(__file__)) + foldername
# Create list of stations in folder
stations = [os.path.basename(x) for x in glob.glob(folderpath + "*_merged.csv")]
stations = [s.replace('_merged.csv', '') for s in stations]


def plot_station(dataframe, stations, folderpath):
    # Create folder for plot files to live
    plt_filename = folderpath + "//Plots//" + str(stations) + "_plot.png"
    if not os.path.exists(folderpath + "//Plots"):
        os.makedirs(folderpath + "//Plots//")
    comp_plot = dataframe[['Observed Flow', 'Predicted Flow']].plot(legend=True, color=['blue', 'green'])
    comp_plot.set_title('Observed and Predicted Flow for ' + stations)
    comp_plot.set_ylabel('Flow (cms)')
    plt.savefig(plt_filename)


def correlation_stats(df, adjustment, beg, end):
    # create Dataframe columns
    cor_coeff = {}
    mean_diff = {}
    mean_var = {}
    month_predicted = {}
    month_observed = {}
    # Monthly metrics
    for i in range(beg, end):
        log_month_predicted = df.where(df.date.dt.month == i).dropna()['log Predicted']
        log_month_observed = df.where(df.date.dt.month == i).dropna()['log Recorded']
        cor_coeff[i] = hs.acc(log_month_observed, log_month_predicted)
        mean_diff[i] = np.exp(scipy.stats.gmean(log_month_observed) - scipy.stats.gmean(log_month_predicted))
        mean_var[i] = np.var(log_month_observed - log_month_predicted)
        month_predicted[i] = np.exp(np.mean(log_month_predicted - adjustment))
        month_observed[i] = np.exp(np.mean(log_month_observed - adjustment))
    # Yearly Metrics
    log_pred = df.dropna()['log Predicted']
    log_observed = df.dropna()['log Recorded']
    cor_coeff[14] = hs.acc(log_observed, log_pred)
    mean_diff[14] = np.exp(scipy.stats.gmean(log_observed) - scipy.stats.gmean(log_pred))
    mean_var[14] = np.var(log_observed - log_pred)
    month_predicted[14] = np.exp(np.mean(log_pred - adjustment))
    month_observed[14] = np.exp(np.mean(log_observed - adjustment))

    # Define dataframe columns
    cor_coeff_df = pd.DataFrame.from_dict(cor_coeff, orient='Index')
    mean_df = pd.DataFrame.from_dict(mean_diff, orient='Index')
    variance_df = pd.DataFrame.from_dict(mean_var, orient='Index')
    predicted_df = pd.DataFrame.from_dict(month_predicted, orient='Index')
    observed_df = pd.DataFrame.from_dict(month_observed, orient='Index')
    correlation_df = pd.concat([cor_coeff_df, mean_df, observed_df, predicted_df, variance_df], axis=1)
    # print(correlation_df)
    correlation_df.columns = ['Correlation', 'Mean Difference', 'Observed Flow', 'Predicted Flow',
                              'Mean Variance']
    return correlation_df


def error_stats(df, beg, end):
    rmse = {}
    rmse_log = {}
    NS_eff = {}
    R2 = {}
    sa = {}
    # calculate metrics by month
    for i in range(beg, end):
        predicted = df.where(df.date.dt.month == i).dropna()['predicted streamflow']
        observed = df.where(df.date.dt.month == i).dropna()['recorded streamflow']
        log_month_predicted = df.where(df.date.dt.month == i).dropna()['log Predicted']
        log_month_observed = df.where(df.date.dt.month == i).dropna()['log Recorded']
        n = predicted.count()
        # print n
        rmse[i] = hs.rmse(predicted, observed)
        rmse_log[i] = hs.rmsle(log_month_predicted, log_month_observed)
        NS_eff[i] = hs.E(log_month_predicted, log_month_observed)
        R2[i] = hs.r_squared(log_month_predicted, log_month_observed)
        sa[i] = hs.sa(predicted, observed)
    # Calculate metrics for year
    predicted = df.dropna()['predicted streamflow']
    observed = df.dropna()['recorded streamflow']
    log_predicted = df.dropna()['log Predicted']
    log_observed = df.dropna()['log Recorded']
    n = predicted.count()
    rmse[14] = hs.rmse(predicted, observed)
    rmse_log[14] = hs.rmsle(log_predicted, log_observed)
    NS_eff[14] = hs.E(log_predicted, log_observed)
    R2[14] = hs.r_squared(predicted, observed)
    sa[14] = hs.sa(predicted, observed)

    rmse_df = pd.DataFrame.from_dict(rmse, orient='Index')
    rmse_log_df = pd.DataFrame.from_dict(rmse_log, orient='Index')
    NS_eff_df = pd.DataFrame.from_dict(NS_eff, orient='Index')
    R2_df = pd.DataFrame.from_dict(R2, orient='Index')
    sa_df = pd.DataFrame.from_dict(sa, orient='Index')
    error_df = pd.concat([rmse_df, rmse_log_df, NS_eff_df, R2_df, sa_df], axis=1)

    error_df.columns = ['RMSE', 'Log RMSE', 'Nash-Sutcliffe Efficiency', 'R^2 Coefficient', 'Spectral Angle']
    return error_df


def monthly_stats_analysis(stations, folderpath, datapath):
    df = pd.read_csv(datapath)
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

    # Sort by Months
    df['date'] = pd.to_datetime(df['Datetime'])
    df.drop('Datetime', axis=1, inplace=True)
    df['difference'] = df[['log Recorded']].sub(df['log Predicted'], axis=0)

    # Correlation Statistics
    # Months are from 1 to 12, 13 is to calculate yearly statistics
    correlation_df = correlation_stats(df, adjustment, 1, 13)
    # Error Metrics
    error_df = error_stats(df, 1, 13)

    # Combine into one dataframe
    results_df = pd.concat([correlation_df, error_df], axis=1)
    results_df.index.name = "Month"
    results_df['Station'] = stations
    print(results_df)

    # Plot Results
    plot_station(results_df, stations, folderpath)

    # Average values
    av_cor = results_df['Correlation'].mean()
    av_mean = results_df['Mean Difference'].mean()
    av_var = results_df['Mean Variance'].mean()
    pred_value = results_df['Predicted Flow']
    obs_value = results_df['Observed Flow']
    rmse_val = hs.rmse(pred_value, obs_value)
    print(av_cor, av_mean, av_var, rmse_val)

    # Print Results
    datapath_results = folderpath + "//Results//" + stations + "_results.csv"
    if not os.path.exists(folderpath + "//Results"):
        os.makedirs(folderpath + "//Results")
    results_df.to_csv(datapath_results, sep=',', index_label="Month")


def combine_csvs(folderpath):
    # Creates summary csv files for each folder of merged datafiles
    results = glob.glob(folderpath + "*_results.csv")
    df_list = []
    for file in tqdm(sorted(results)):
        df_list.append(pd.read_csv(file))
    df_results = pd.concat(df_list)
    df_results.to_csv(folderpath + 'National Results.csv', sep=',', index=False)


# Run statistics for each station in station list
for i in stations:
    station = str(i)
    print(station)
    datapath = str(folderpath) + station + "_merged.csv"
    monthly_stats_analysis(station, folderpath, datapath)

# Combine results file to summary file
combine_csvs(folderpath + "//Results//")
