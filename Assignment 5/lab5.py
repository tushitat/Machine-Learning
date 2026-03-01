import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def load_dataset():
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "DCT_mal (2) 1.csv")
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def perform_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    return kmeans


def clustering_scores(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, ch, db


if __name__ == "__main__":

    X, y = load_dataset()
    y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train_single = X_train[:, [0]]
    X_test_single = X_test[:, [0]]

    model_single = train_linear_regression(X_train_single, y_train)
    y_train_pred_single = model_single.predict(X_train_single)
    y_test_pred_single = model_single.predict(X_test_single)

    mse_tr, rmse_tr, mape_tr, r2_tr = regression_metrics(y_train, y_train_pred_single)
    mse_te, rmse_te, mape_te, r2_te = regression_metrics(y_test, y_test_pred_single)

    print("A1-A2 Single Feature Train Metrics:", mse_tr, rmse_tr, mape_tr, r2_tr)
    print("A1-A2 Single Feature Test Metrics:", mse_te, rmse_te, mape_te, r2_te)

    model_multi = train_linear_regression(X_train, y_train)
    y_train_pred_multi = model_multi.predict(X_train)
    y_test_pred_multi = model_multi.predict(X_test)

    mse_tr_m, rmse_tr_m, mape_tr_m, r2_tr_m = regression_metrics(y_train, y_train_pred_multi)
    mse_te_m, rmse_te_m, mape_te_m, r2_te_m = regression_metrics(y_test, y_test_pred_multi)

    print("A3 Multi Feature Train Metrics:", mse_tr_m, rmse_tr_m, mape_tr_m, r2_tr_m)
    print("A3 Multi Feature Test Metrics:", mse_te_m, rmse_te_m, mape_te_m, r2_te_m)

    kmeans_2 = perform_kmeans(X_train, 2)
    print("A4 Cluster Centers (k=2):", kmeans_2.cluster_centers_)

    sil, ch, db = clustering_scores(X_train, kmeans_2.labels_)
    print("A5 Scores (k=2):", sil, ch, db)

    sil_scores = []
    ch_scores = []
    db_scores = []
    distortions = []

    k_range = range(2, 11)

    for k in k_range:
        kmeans = perform_kmeans(X_train, k)
        sil, ch, db = clustering_scores(X_train, kmeans.labels_)
        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        distortions.append(kmeans.inertia_)

    plt.plot(k_range, sil_scores)
    plt.title("Silhouette Score vs k")
    plt.show()

    plt.plot(k_range, ch_scores)
    plt.title("Calinski-Harabasz Score vs k")
    plt.show()

    plt.plot(k_range, db_scores)
    plt.title("Davies-Bouldin Score vs k")
    plt.show()

    plt.plot(k_range, distortions)
    plt.title("Elbow Plot")
    plt.show()