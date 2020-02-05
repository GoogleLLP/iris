import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("iris.csv", header=None)
    # 计算每种花的数量，并画出柱状图
    y_val = df.iloc[:, -1].value_counts().values
    x_val = df.iloc[:, -1].value_counts().index.values
    plt.figure(1)
    plt.title("number of each kind of flower")
    plt.xlabel("flower kind")
    plt.ylabel("flower number")
    plt.grid(True)
    plt.xticks(x_val)
    plt.bar(x_val, y_val)
    plt.show()

    # 把4个特征做正则化处理
    norm = lambda x: (x - x.min())/(x.max() - x.min())
    X = df.iloc[:, 0:-1].apply(norm)
    print(X.head())

    # 把数据75%的数据当成训练集，25%的数据作为测试集，使用KNN分类模型，k=2，根据训练集，获得模型，并输出accuracy
    X = X.values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)
    print("accuracy:%s" % accuracy_score(model.predict(X_test), y_test))

    # 在训练集上，使用5-fold cross validation调整K的值，选择accuracy效果最好的K值
    results = []
    Ks = range(2, 30)
    for k in Ks:
        model = KNeighborsClassifier(n_neighbors=k)
        results.append(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")))
    plt.figure(2)
    plt.title("accuracy of KNN")
    plt.xlabel("K value")
    plt.ylabel("accuracy")
    plt.xticks(Ks)
    plt.grid(True)
    plt.plot(Ks, results, "r-o")
    plt.show()
    idx = np.where(np.max(results) == results)[0].tolist()
    Ks = np.array(Ks)
    print("最好的K值是%d" % Ks[idx[0]])

    # 使用最好的K值训练KNN分类模型，并在测试集上获得accuracy结果
    model = KNeighborsClassifier(n_neighbors=Ks[idx[0]])
    model.fit(X_test, y_test)
    print("对应的accuracy是%s" % accuracy_score(model.predict(X_test), y_test))
