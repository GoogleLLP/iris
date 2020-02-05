import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 任务1
    df = pd.read_csv("iris.csv", header=None)
    X = df.values[:, 0:-1]
    y = df.values[:, -1]
    corr_pearson = []
    for i in range(np.size(X, 1)):
        corr_pearson.append(pearsonr(X[:, i], y))
    print(corr_pearson)

    # 任务2
    model = KMeans(n_clusters=2)
    model.fit(X)

    # 任务3
    Ks = [2, 3, 4, 5, 6, 7]
    loss = []
    euclidean = []
    for k in Ks:
        model = KMeans(n_clusters=k, n_jobs=-1)
        model.fit(X)
        loss.append(-model.score(X))
        euclidean.append(silhouette_score(X, model.labels_, metric="euclidean"))
    K = Ks[np.argmax(euclidean)]
    print(K)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.xlabel("K")
    plt.ylabel("loss score")
    plt.xticks(Ks)
    plt.plot(Ks, loss, "r-o")
    plt.subplot(1, 2, 2)
    plt.title("euclidean")
    plt.xlabel("K")
    plt.ylabel("euclidean")
    plt.xticks(Ks)
    plt.plot(Ks, euclidean, "b-o")
    plt.show()

    # 任务4
    corr = np.array(list(zip(*corr_pearson))[0])
    index = np.argsort(-corr)[0:2]
    X = X[:, index]
    plt.figure(2)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    # 任务5
    model = KMeans(n_clusters=K)
    model.fit(X)
    labels = model.labels_
    plt.figure(3)
    for i in range(model.n_clusters):
        points = X[labels == i, :]
        plt.scatter(points[:, 0], points[:, 1], marker=(i+1, 1))
    plt.show()


