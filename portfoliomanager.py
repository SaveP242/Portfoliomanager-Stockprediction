import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Ml_model(dataset):
    dataset_train = dataset.truncate(before='2020-01-01', after='2022-01-01')
    dataset_train = dataset_train.resample('D').fillna(method='bfill')
    print(dataset_train.head())
    print(dataset_train.isna().any())
    print(dataset_train.info())
    print(dataset_train.rolling(7).mean().head(12))
    ts = dataset_train["Open"]
    ts = pd.DataFrame(ts)
    n = len(ts)
    print(n)
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    tss = sc.fit_transform(ts)
    X_train = []
    y_train = []
    for i in range(60, n):
        X_train.append(tss[i - 60:i, 0])
        y_train.append(tss[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    r = Sequential()
    r.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # r.add(Dropout(0.2))
    r.add(LSTM(units=50, return_sequences=True))
    # r.add(Dropout(0.2))
    r.add(LSTM(units=50, return_sequences=True))
    # r.add(Dropout(0.2))
    r.add(LSTM(units=50))
    # r.add(Dropout(0.2))
    r.add(Dense(units=1))

    r.compile(optimizer='adam', loss="mean_squared_error")
    history = r.fit(X_train, y_train, epochs=1, batch_size=16)

    dataset_test = dataset.truncate(before='2022-01-01', after='2022-02-01')
    dataset_test = dataset_test.resample('D').fillna(method='bfill')

    real_price = dataset_test.iloc[:, 1:2].values

    test_set = dataset_test['Open']
    test_set = pd.DataFrame(test_set)
    m = len(test_set)

    total_dataset = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = total_dataset[len(total_dataset) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 60 + m):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    p = r.predict(X_test)
    p = sc.inverse_transform(p)
    p = pd.DataFrame(p)
    p = p.reset_index(drop=True)

    p_max = p.max()
    p_max = np.array(p_max)
    p_max = p_max[0]
    print(p_max)
    max_arr = p.idxmax().values
    max_idx = max_arr[0]
    print(max_idx)
    p_max1= p.head(max_idx).max()


    min_set = p.head(max_idx)
    p_min = min_set.min()
    p_min = np.array(p_min)
    p_min = p_min[0]
    print(p_min)
    min_arr = min_set.idxmin().values
    min_idx = min_arr[0]
    print(min_idx)

    profit = p_max - p_min
    print(profit)

    real_min = real_price[min_idx]
    real_max = real_price[max_idx]

    real_profit = real_max - real_min

    data_list = [profit, p_min, real_profit]

    # plt.plot(p, color='blue')
    # plt.plot(real_price, color='red')
    # plt.show()

    return data_list


a=pd.read_csv("TCS.BO.csv",index_col="Date",parse_dates=True)
b=pd.read_csv("HCLTECH.BO.csv",index_col="Date",parse_dates=True)
c=pd.read_csv("INFY.BO.csv",index_col="Date",parse_dates=True)
d=pd.read_csv("LTI.BO.csv",index_col="Date",parse_dates=True)
e=pd.read_csv("WIPRO.BO.csv",index_col="Date",parse_dates=True)

a_arr= Ml_model(a)
b_arr= Ml_model(b)
c_arr= Ml_model(c)
d_arr= Ml_model(d)
e_arr= Ml_model(e)

a_profit= a_arr[0].astype(int)
b_profit= b_arr[0].astype(int)
c_profit= c_arr[0].astype(int)
d_profit= d_arr[0].astype(int)
e_profit= e_arr[0].astype(int)

a_buy= a_arr[1].astype(int)
b_buy= b_arr[1].astype(int)
c_buy= c_arr[1].astype(int)
d_buy= d_arr[1].astype(int)
e_buy= e_arr[1].astype(int)

a_real= a_arr[2]
b_real= b_arr[2]
c_real= c_arr[2]
d_real= d_arr[2]
e_real= e_arr[2]

#Dynamic Programming to maximise profits

def max_profit(w, n, price, profit, real):
    dp= [[0] * (w+1) for i in range (n)]

    for i in range(n):
        dp[i][0]

    for j in range(w+1):
        dp[0][j]= (j//price[0] * profit[0])

    for i in range (1, n):
        for j in range(1, w+1):
            if (price[i] <= j):
                dp[i][j]= max(dp[i-1][j], dp[i][j- price[i]] +profit[i])
            else:
                dp[i][j] = dp[i-1][j]

    get_stocks(dp, price, profit, real)
    return dp[i][j]

def get_stocks(dp, price, profit, real):
    i,j= len(dp)-1, len(dp[0])-1
    total_profit= dp[i][j]
    x=0
    print("Stocks to buy: ")
    while(i-1>= 0 and dp[i][j]>= 0):
        if(dp[i][j] != dp[i-1][j]):
            if(i==0):
                print("TCS")
                x += real[0]
            elif(i==1):
                print("HCL")
                x += real[1]
            elif(i==2):
                print("INFOSYS")
                x += real[2]
            elif(i==3):
                print("LTI")
                x += real[3]
            else:
                print("WIPRO")
                x += real[4]
            j -= price[i]
            total_profit -= profit[i]
        else:
            i -= 1

    while(total_profit > 1):
        print("TCS")
        x+= real[0]
        total_profit -= profit[0]

    print(x)

profit_arr= [a_profit, b_profit, inf_profit, lti_profit, wip_profit]

buy_arr= [a_buy, b_buy, inf_buy, lti_buy, wip_buy]

real_arr= [tcs_real, hcl_real, inf_real, lti_real, wip_real]

print(type(profit_arr))
print(type(buy_arr))

wallet= input("Enter the amount: ")
wallet= int(wallet)

size= len(buy_arr)

print(max_profit(wallet, size, buy_arr, profit_arr, real_arr))
print(real_arr)