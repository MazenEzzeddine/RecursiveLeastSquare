import numpy as np
import math


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import pandas_datareader as web
import datetime

from hisRLS import RLS
from myRLS import myRLS

register_matplotlib_converters()
import datetime






def iceCream():
    df_ice_cream = pd.read_csv('ice_cream.csv')


    print(df_ice_cream.head())
    df_ice_cream.rename(columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)
    df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
    df_ice_cream.set_index('date', inplace=True)
    start_date = pd.to_datetime('2005-01-01')
    df_ice_cream = df_ice_cream[start_date:]
    print(df_ice_cream.head())

    y = pd.DataFrame(df_ice_cream)
    ##j = y['production'].values
    print(y)
    df = y['production'].values
    # print(df)
    test_size = len(df)
    lam = 0.98
    num_vars = 100
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
            if (i > num_vars):
                # intial state, update
                x[0, j] = df[i - j -1]  # or inverse
            else:
                x[0, j] = 0
        pred_x.append(i)
        pred_y.append(float(x * LS.w))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, df[i])
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], df[0:], label='actual')
    plt.title("ice cream production prediction")
    #########################################################
    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')

    plt.legend()
    plt.show()

# # residual plot
#     _ = plt.plot(pred_x [50:], df[50:] - pred_y[50:], label='residual')
#     plt.show()



def seriesTest():
    start = datetime.date(2016, 7, 11)
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

    x = np.matrix(np.zeros((1, num_vars)))

    for i in range(test_size):
        print(y[i])
        # for j in range(num_vars):
        #    # x[0, j] = i ** j
        #    if (i > 5):
        #        x[0, j] = y[i - j - 2]  # or inverse
        #        x[0, j] = y[i - j - 2]
        #
        #    else:
        #        x[0, j] = 0

        for j in range(num_vars):
            if (i > 0):
                x[0, j - 1] = x[0, j]
            else:
                x[0, j] = y[i]
        pred_x.append(i)
        pred_y.append(float(x * LS.w))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, y[i])
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], y[0:], label='actual')
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
        # for j in range(5):
        #     if(i>5):
        #         x[0, j] = y[i-j] #or inverse
        #     else:
        #         x[0, j] = 0

        # for j in range(num_vars):
        #     if (i > 0):
        #         x[0, j - 1] = x[0, j]
        #     else:
        #         x[0, j] = df[i]

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

def iceCream2():
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
    num_vars = 1
    LS = myRLS(num_vars, lam)
    # We won't use RLS.fit because I want to save the predictions.
    pred_x = []
    pred_y = []

    x = np.matrix(np.zeros((1, num_vars)))


    for j in range(num_vars):
        x[0, j] = 0

    pred_error = []
    for i in range(test_size):
        # print(y[i])

        # for j in range(num_vars):
        #     if(j<num_vars-1):
        #         x[0, j] =  x[0, j+1]
        #     else:
        #         x[0, 0] = df[i]
        #
        # for j in range(num_vars):
        #     if i > 1:
        #         x[0, j-1] = x[0, j]
        #     else:
        #         x[0, j] = df[i]

        ###########################################


        # for j in range(num_vars):
        #     x[0, j] = x[0, j-1]
        #
        # x[0, 0] = df[i]



        for j in range(num_vars):
            x[0, j-1] = x[0, j]

        x[0, j] = df[i]


        pred_x.append(i)
        pred_y.append(float(x * LS.w))

        print("{}, {}", float(x * LS.w), float(df[i]))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, df[i])
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], df[0:], label='actual')
    plt.title("ice cream production prediction")
    #########################################################
    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')

    plt.legend()
    plt.show()



def iceCream4():
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
    num_vars = 1
    LS = RLS(num_vars, lam, 10)
    # We won't use RLS.fit because I want to save the predictions.
    pred_x = []
    pred_y = []

    x = np.matrix(np.zeros((1, num_vars)))


    for j in range(num_vars):
        x[0, j] = 0

    pred_error = []
    for i in range(test_size):
        # print(y[i])

        # for j in range(num_vars):
        #     if(j<num_vars-1):
        #         x[0, j] =  x[0, j+1]
        #     else:
        #         x[0, 0] = df[i]
        #
        # for j in range(num_vars):
        #     if i > 1:
        #         x[0, j-1] = x[0, j]
        #     else:
        #         x[0, j] = df[i]

        ###########################################


        # for j in range(num_vars):
        #     x[0, j] = x[0, j-1]
        #
        # x[0, 0] = df[i]

        for j in reversed(range(num_vars)):
            if j == 0:
                continue
            x[0, j] = x[0, j - 1]
        x[0, 0] = df[i]
        pred_x.append(i)
        pred_y.append(float(x * LS.w))





        print("{}, {}", float(x * LS.w), float(df[i]))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, df[i])
    ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    _ = plt.plot(pred_x[0:], df[0:], label='actual')
    plt.title("ice cream production prediction")
    #########################################################
    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')

    plt.legend()
    plt.show()



###############################################################################################

def iceCream3():
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
    num_vars = 4
    #LS = RLS(num_vars, lam, 10)
    LS = myRLS(num_vars, lam)
    # We won't use RLS.fit because I want to save the predictions.
    pred_x = []
    pred_y = []

    x = np.matrix(np.zeros((1, num_vars)))


    for j in range(num_vars):
        x[0, j] = 0

    pred_error = []
    for i in range(90):

        for j in reversed(range(num_vars)):
            if j == 0:
                continue
            x[0, j] = x[0, j - 1]
        x[0, 0] = df[i]
        pred_x.append(i)
        pred_y.append(float(x * LS.w))





        print("{}, {}", float(x * LS.w), float(df[i]))
        pred_error.append(LS.get_error())
        LS.add_obs(x.T, df[i])
    # ax = plt.plot(pred_x[0:], pred_y[0:], label='predicted')
    # _ = plt.plot(pred_x[0:], df[0:], label='actual')
    # plt.title("ice cream production prediction")
    #########################################################
    # ax = plt.plot(pred_x[1:], pred_y[1:], label='predicted')
    # _ = plt.plot(pred_x[1:], y[1:], label='actual')

    # plt.legend()
    # plt.show()

    pred_xt = []
    pred_yt = []

    #for j in range(num_vars):
    x[0, 0] = df[89]
    x[0, 1] = df[88]
    x[0, 2] = df[87]
    x[0, 3] = df[86]



    for i in range(90, test_size):
        for j in reversed(range(num_vars)):
            if j == 0:
                continue
            x[0, j] = x[0, j - 1]
        pred_xt.append(i)
        pred_yt.append(float(x * LS.w))
        x[0, 0] = df[i]


    print(len(pred_xt))
    print(len(pred_yt))
    print(len(df[90:test_size ]))


    ax = plt.plot(pred_xt, pred_yt, label='predicted')
    _ = plt.plot(pred_xt, df[90:test_size], label='actual')
    plt.title("ice cream production prediction")
    plt.legend()
    plt.show()









if __name__ == '__main__':
    #firstTest()
    #seriesTest()
    #iceCream2()
    iceCream3()
    #iceCream4()





