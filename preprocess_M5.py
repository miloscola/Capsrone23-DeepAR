from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange
import datetime as dt


from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prep_data(data, covariates, data_start, train = True):

    print("train: ", train)
    time_len = data.shape[0] # time_len = 1913
    
    print("time_len: ", time_len)
    input_size = window_size-stride_size # input_size = 192-24 = 168
    
    #每個item有幾個window
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size) #(193-168)//24
    print("windows pre: ", windows_per_series.shape)

    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    print("data_start: ", data_start.shape)
    print(data_start)
    print("windows: ", windows_per_series.shape)
    print(windows_per_series)

    # bug is from here!!!!!!
    ##################################################
    # the original way to count windows_per_series and sum(count windows_per_series[series]) is inconsistant
    # so i recalculate it



    # new one
    total_windows = 0
    for series in trange(num_series):
        for i in range(windows_per_series[series]):
            total_windows += 1
    print("total_windows: ", total_windows)
    ##################################################

    # if not train:
    #     total_windows = np.sum(windows_per_series) 

    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    print("x_input: ", x_input.shape)
    print("label: ", label.shape)
    print("v_input: ", v_input.shape)
    print(num_series)

    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    
    
    # k = 0
    # m = 0
    # print(windows_per_series[0])
    # print(windows_per_series[1])
    # print(windows_per_series[2])
    # print(windows_per_series[3])
    # for series in trange(num_series):
    #     # print(sum((windows_per_series[series])))
    #     m += windows_per_series[series]
    #     for i in range(windows_per_series[series]):
    #         k += 1
    # print('m = ', m)
    # print('k = ', k)
    
    count = 0
    print('covariates1:', covariates[:,0].shape)
    if not train:
        covariates = covariates[-time_len:]
        print('covariates2:', covariates.shape)
    print('covariates3:', covariates.shape)
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if(series == 1):
            print('cov_age:',cov_age.shape)
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
            
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            
            # if(count > 100000):
            #     print("x: ", x_input[count, 1:, 0].shape)
            #     print("window start: ", window_start)
            #     print("window end: ", window_end)
            #     print("data: ", data.shape)
            #     print("d: ", data[window_start:window_end-1, series].shape)
            
            try:
                x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            except:
                print(count, window_start, window_end, series)
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1

    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+ save_name, x_input)
    np.save(prefix+'v_'+ save_name, v_input)
    np.save(prefix+'label_'+ save_name, label)

"""
    生成协变量
    times：时间戳（行数）
    num_covariates：协变量的数量（列数）
"""
# def gen_covariates(times, num_covariates):
#     covariates = np.zeros((times.shape[0], num_covariates))
#     for i, input_time in enumerate(times):
#         covariates[i, 1] = input_time.weekday()
#         covariates[i, 2] = input_time.hour
#         covariates[i, 3] = input_time.month
#     for i in range(1,num_covariates):
#         covariates[:,i] = stats.zscore(covariates[:,i])
#     return covariates[:, :num_covariates]

def gen_m5_covariates(calendar_df):
    # Assuming calendar_df is the calendar DataFrame and sales_df is the sales data DataFrame
    num_covariates = 6  # Example: day of week, month, SNAP_CA, SNAP_TX, SNAP_WI, event
    times = pd.to_datetime(calendar_df['date'])
    covariates = np.zeros((len(times), num_covariates))
    # Day of week and month as numerical values
    covariates[:, 0] = times.dt.dayofweek
    covariates[:, 1] = times.dt.month
    # SNAP days for CA, TX, and WI
    covariates[:, 2] = calendar_df['snap_CA']
    covariates[:, 3] = calendar_df['snap_TX']
    covariates[:, 4] = calendar_df['snap_WI']
    # Event (simple binary indicator for this example)
    covariates[:, 5] = calendar_df[['event_name_1', 'event_name_2']].notnull().any(axis=1).astype(int)
    # Standardize covariates (optional depending on model requirements)
    for i in range(num_covariates):
        covariates[:, i] = (covariates[:, i] - covariates[:, i].mean()) / covariates[:, i].std()
    return covariates

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))


if __name__ == '__main__':

    global save_path
    name_sales = 'sales_train_evaluation.csv'
    name_calendar = 'calendar.csv'
    name_prices = 'sell_prices.csv'
    save_name = 'M5'
    window_size = 56
    stride_size = 7
    num_covariates = 6
    train_start = '2011-01-29 00:00:00'
    train_end = '2016-04-24 00:00:00'
    test_start = '2016-03-01 00:00:00' #need additional 7 days as given info
    test_end = '2016-05-22 00:00:00'
    pred_days = 28
    # given_days = 7

    save_path = os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sales_csv_path = os.path.join(save_path, name_sales)
    calendar_csv_path = os.path.join(save_path, name_calendar)
    prices_csv_path = os.path.join(save_path, name_prices)

    sales = pd.read_csv(sales_csv_path)
    sales.name = 'sales'
    calendar = pd.read_csv(calendar_csv_path)
    calendar.name = 'calendar'
    prices = pd.read_csv(prices_csv_path)
    prices.name = 'prices'

    #Create date index
    date_index = calendar['date']
    dates = date_index[0:1941]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

    # Create a data frame for items sales per day with item ids (with Store Id) as columns names  and dates as the index 
    sales['item_store_id'] = sales.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)
    data_frame = sales.loc[:,'d_1':'d_1941'].T
    data_frame.columns = sales['item_store_id'].values

    #Set Dates as index 
    data_frame = pd.DataFrame(data_frame).set_index([dates_list])
    data_frame.index = pd.to_datetime(data_frame.index)


    #Set Dates as index 
    data_frame = pd.DataFrame(data_frame).set_index([dates_list])
    data_frame.index = pd.to_datetime(data_frame.index)

    covariates = gen_m5_covariates(calendar)
    train_data = data_frame[train_start:train_end].values
    test_data = data_frame[test_start:test_end].values
    data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
    total_time = data_frame.shape[0] #1941 (days)
    num_series = data_frame.shape[1] #30490 (item_id)

    prep_data(train_data, covariates, data_start)
    print('len test data:',len(test_data))
    prep_data(test_data, covariates, data_start, train=False)
