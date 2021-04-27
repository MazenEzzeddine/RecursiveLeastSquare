


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import pandas_datareader as web
import datetime
from myRLS import myRLS

register_matplotlib_converters()
import datetime


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    n = 5  # number of points
    x = np.linspace(0, 1, n)  # n points in [0, 1]
    y = np.zeros(n)  # n zeros (float data type)
    for i in range(n):
        y[i] = f(x[i])
        print(y[i])



def iceCream():
    df_ice_cream = pd.read_csv('ice_cream.csv')


    print(df_ice_cream.head())
    df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)
    df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
    df_ice_cream.set_index('date', inplace=True)
    start_date = pd.to_datetime('2010-01-01')
    df_ice_cream = df_ice_cream[start_date:]
    print(df_ice_cream.head())

    y = pd.DataFrame(df_ice_cream)
    ##j = y['production'].values
    print(y)
    df = y['production'].values
    # print(df)
    test_size = len(df)
    lam = 0.98
    delta = 10
    num_vars = 8
    LS = myRLS(num_vars, lam)
    # We won't use RLS.fit because I want to save the predictions.
    pred_x = []
    pred_y = []
    pred_error = []
    for i in range(test_size):
        # print(y[i])
        x = np.matrix(np.zeros((1, num_vars)))
        for j in range(num_vars):
            #x[0, j] = i ** j
            if (i > 8):
                x[0, j] = df[i - j]  # or inverse
            else:
                x[0, j] = i ** j
        pred_x.append(i)
        pred_y.append(float(x * LS.w))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, df[i])
    ax = plt.plot(pred_x[50:], pred_y[50:], label='predicted')
    _ = plt.plot(pred_x[50:], df[50:], label='actual')
    plt.title("ice cream production prediction")
    #########################################################
    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')
    # _ = plt.title("RLS nasrallah")
    plt.legend()
    plt.show()

# # residual plot
#     _ = plt.plot(pred_x [50:], df[50:] - pred_y[50:], label='residual')
#     plt.show()



def seriesTest():
    start = datetime.date(2014, 7, 11)
    end = datetime.date(2021, 3, 7)
    f = web.DataReader('SPY', 'iex', start, end, api_key='pk_f50c1a4af6cc468a9fd0d853f0a5478c')


    # pk_f50c1a4af6cc468a9fd0d853f0a5478c
    test_size = len(f)
    y = f['close'].values
    print(y)
    lam = 0.98
    delta = 10
    num_vars = 5
    LS = myRLS(num_vars, lam)
    # We won't use RLS.fit because I want to save the predictions.
    pred_x = []
    pred_y = []
    pred_error = []
    for i in range(test_size):
        print(y[i])
        x = np.matrix(np.zeros((1, num_vars)))
        for j in range(num_vars):
           # x[0, j] = i ** j
           if (i > 5):
               x[0, j] = y[i - j]  # or inverse
           else:
               x[0, j] = i ** j
        pred_x.append(i)
        pred_y.append(float(x * LS.w))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, y[i])
    #ax = plt.plot(pred_x[50:], pred_y[50:], label='predicted')
    _ = plt.plot(pred_x[50:], y[50:], label='actual')
    plt.title("SPY stock indicator closing price, 11/7/2014 - 3/7/2021")

    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')
    # _ = plt.title("RLS nasrallah")
    plt.legend()
    plt.show()



def firstTest():
    test_size = 3000
    # Test function
    f = lambda x: 0.2 * x ** 3 - 3.8 * x** 2 - 5.1

    noise = [0 for i in range(test_size)]

    y = np.array([f(i) for i in range(test_size)])
    noisy_y = y + noise
    lam = 0.98
    LS = myRLS(5, lam)
    # Not using the RLS.fit function because I want to remember all the predicted values
    pred_x = []
    pred_y = []
    for i in range(test_size):
        x = np.matrix(np.zeros((1, 5)))
        for j in range(5):
            if(i>5):
                x[0, j] = y[i-j] #or inverse
            else:
                x[0, j] = i ** j

        pred_x.append(i)
        pred_y.append(float(x * LS.w))
        LS.add_obs(x.T, y[i])
    # LS.add_obs(x.T, y[i])
    print(LS.w)
    # plot the predicted values against the non-noisy output
    # ax = plt.plot(pred_x, y - pred_y)
    # # ax = plt.plot(pred_x, y, pred_y)
    # plt.show()
    ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    _ = plt.plot(pred_x[1:], y[1:], label='actual')
    plt.show()




if __name__ == '__main__':
    #firstTest()
    seriesTest()
    #iceCream()





