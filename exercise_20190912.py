# %% 导入包
import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt


# %% 定义损失函数J关于theta的函数
def cost_function(x, y, theta):
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    theta = np.asmatrix(theta)
    y_hat = x * theta
    return (y - y_hat).T * (y - y_hat) / x.shape[0]


# %% 定义一个求偏导的函数
def partial_derivative(func, params, epsilon=1e-3, return_cost=False):
    x, y, theta = params
    cost = func(x, y, theta)
    result = np.asmatrix(np.zeros((1, x.shape[1])))
    # 求各个方向梯度
    for i in range(x.shape[1]):
        temp = theta.copy()
        temp[i, 0] = temp[i, 0] + epsilon
        cost_plus = func(x, y, temp)
        result[0, i] = ((cost_plus - cost) / epsilon)[0, 0]
    if return_cost:
        return result, cost[0, 0]
    else:
        return result


# %% 定义最小二乘函数
def least_squares(x, y):
    x = np.asmatrix(x)
    y = np.asmatrix(y.reshape(-1, 1))
    return inv(x.T * x) * x.T * y


# %% 梯度下降
# 参考：https://www.cnblogs.com/pinard/p/5970503.html
def gradient_descent(x, y, n_iter=1000, alpha=0.01, learning_curve=False):
    cost_curve = []
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    theta = np.ones(x.shape[1])
    theta = np.asmatrix(theta)
    for i in range(n_iter):
        cost_curve.append(((theta * x.T - y) * (theta * x.T - y).T / x.shape[0])[0, 0])
        # Σx²等价于X.T*X
        theta = theta - alpha * 2 / x.shape[0] * (theta * x.T - y) * x
    if learning_curve:
        plt.figure()
        plt.xlabel("n_iter")
        plt.ylabel("cost")
        plt.plot(range(len(cost_curve)), cost_curve)
        plt.show()
    return theta


# %% 通过伪梯度实现梯度下降
def pseudo_gradient_descent(x, y, n_iter=1000, alpha=1e-5, learning_curve=False):
    cost_curve = []
    y = y.reshape(-1, 1)
    x = np.asmatrix(x)
    y = np.asmatrix(y)
    theta = np.ones((x.shape[1], 1))
    theta = np.asmatrix(theta)
    for i in range(n_iter):
        pseudo_gradient, cost = partial_derivative(cost_function, (x, y, theta), return_cost=True)
        cost_curve.append(cost)
        theta = theta - alpha * pseudo_gradient.T
    if learning_curve:
        plt.figure()
        plt.xlabel("n_iter")
        plt.ylabel("cost")
        plt.plot(range(len(cost_curve)), cost_curve)
        plt.show()
    return theta


# %%
if __name__ == "__main__":
    # %% 加载数据，并用自带线性回归分析
    X = load_iris().data
    y = load_iris().target
    model = LinearRegression()
    model.fit(X, y)
    print(mean_squared_error(y, model.predict(X)))

    # %% 用最小二乘预测
    w = least_squares(X, y)
    X_pred_ls = X * w
    print(mean_squared_error(y, X_pred_ls))

    # %% 使用梯度下降预测
    w = gradient_descent(X, y, learning_curve=True)
    X_pred_gd = X * w.T
    print(mean_squared_error(y, X_pred_gd))

    # %% 使用伪梯度下降预测
    loss = []
    w = pseudo_gradient_descent(X, y, n_iter=2000, alpha=0.01, learning_curve=True)
    X_pred_pgd = np.asmatrix(X) * w
    print(mean_squared_error(y, X_pred_pgd))

    plt.figure(1)
    plt.plot(range(X.shape[0]), model.predict(X).ravel(), "r-")
    plt.plot(range(X.shape[0]), np.asarray(X_pred_ls).ravel(), "g-")
    plt.plot(range(X.shape[0]), np.asarray(X_pred_gd).ravel(), "b-")
    plt.plot(range(X.shape[0]), np.asarray(X_pred_pgd).ravel(), "k:")
    plt.show()





