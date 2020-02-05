from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np


def argmax(lst):
    return lst.index(max(lst))


if __name__ == "__main__":
    # 数据预处理
    X = load_iris().data[:, [0, 2]]
    y = load_iris().target
    X = minmax_scale(X)

    # 决策树
    model = DecisionTreeClassifier(max_depth=4, random_state=1)
    model.fit(X, y)
    xx, yy = np.meshgrid(np.arange(X[:, 0].min(), X[:, 0].max(), 0.01), np.arange(X[:, 1].min(), X[:, 1].max(), 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    print("决策树 accuracy =", model.score(X, y))
    # plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title("Decision Tree")
    cs = plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.grid(False)
    # plt.show()

    # SVM
    X = load_iris().data[:, [0, 2]]
    y = load_iris().target
    model = SVC(gamma="scale", random_state=1)
    model.fit(X, y)
    xx, yy = np.meshgrid(np.arange(X[:, 0].min(), X[:, 0].max(), 0.1), np.arange(X[:, 1].min(), X[:, 1].max(), 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    print("SVM accuracy =", model.score(X, y))
    # plt.figure(2)
    plt.subplot(2, 2, 2)
    plt.title("SVM")
    cs = plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.grid(False)
    # plt.show()

    # KNN
    Ks = range(2, 20)
    models = []
    scores = []
    for K in Ks:
        model = KNeighborsClassifier(n_neighbors=K)
        model.fit(X, y)
        models.append(model)
        scores.append(cross_val_score(model, X, y, cv=5, n_jobs=6, scoring="accuracy").mean())
    K = Ks[argmax(scores)]
    accuracy = scores[argmax(scores)]
    model = models[argmax(scores)]
    print("最佳K值:", K)
    print("KNN accuracy =", accuracy)
    xx, yy = np.meshgrid(np.arange(X[:, 0].min(), X[:, 0].max(), 0.01), np.arange(X[:, 1].min(), X[:, 1].max(), 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # plt.figure(3)
    plt.subplot(2, 2, 3)
    plt.title("KNN")
    cs = plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.grid(False)
    plt.show()





