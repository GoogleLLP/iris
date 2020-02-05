# encoding:utf-8
"""
  Author:   njulpy
  Version:   1.0
  Data:   2018/04/09
  Project： Using Python to Implement LineRegression Algorithm
"""
import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import dot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model


# 最小二乘法
def lms(x_train, y_train, x_test):
    theta_n = dot(dot(inv(dot(x_train.T, x_train)), x_train.T), y_train)  # theta = (X'X)^(-1)X'Y
    # print(theta_n)
    y_pre = dot(x_test, theta_n)
    mse = np.average((y_test - y_pre) ** 2)
    # print(len(y_pre))
    # print(mse)
    return theta_n, y_pre, mse


# 梯度下降算法
def train(x_train, y_train, num, alpha, m, n):
    # num是迭代次数
    beta = np.ones(n)
    for i in range(num):
        h = np.dot(x_train, beta)  # 计算预测值
        error = h - y_train.T  # 计算预测值与训练集的差值
        delt = 2 * alpha * np.dot(error, x_train) / m  # 计算参数的梯度变化值
        beta = beta - delt
        # print('error', error)
    return beta


if __name__ == "__main__":
    iris = pd.read_csv('iris.csv')
    iris['Bias'] = float(1)
    x = iris[['Sepal.Width', 'Petal.Length', 'Petal.Width', 'Bias']]
    y = iris['Sepal.Length']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
    t = np.arange(len(x_test))
    m, n = np.shape(x_train)
    # Leastsquare
    theta_n, y_pre, mse = lms(x_train, y_train, x_test)
    # plt.plot(t, y_test, label='Test')
    # plt.plot(t, y_pre, label='Predict')
    # plt.show()
    # GradientDescent
    beta = train(x_train, y_train, 1000, 0.001, m, n)
    y_predict = np.dot(x_test, beta.T)
    # plt.plot(t, y_predict)
    # plt.plot(t, y_test)
    # plt.show()
    # sklearn
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_p = regr.predict(x_test)
    print(regr.coef_, theta_n, beta)
    l1, = plt.plot(t, y_predict)
    l2, = plt.plot(t, y_p)
    l3, = plt.plot(t, y_pre)
    l4, = plt.plot(t, y_test)
    plt.legend(handles=[l1, l2, l3, l4], labels=['GradientDescent', 'sklearn', 'Leastsquare', 'True'], loc='best')
    plt.show()



