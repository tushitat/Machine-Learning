import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y, data

def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def euclidean_norm(A):
    return (sum(a * a for a in A)) ** 0.5

def mean_vector(X):
    return np.sum(X, axis=0) / X.shape[0]

def variance_vector(X):
    mu = mean_vector(X)
    return np.sum((X - mu) ** 2, axis=0) / X.shape[0]

def std_vector(X):
    return variance_vector(X) ** 0.5

def class_statistics(X, y, cls):
    Xc = X[y == cls]
    return mean_vector(Xc), std_vector(Xc)

def interclass_distance(mu1, mu2):
    return np.linalg.norm(mu1 - mu2)

def feature_histogram(feature):
    hist, bins = np.histogram(feature, bins=10)
    mean = np.mean(feature)
    var = np.var(feature)
    return hist, bins, mean, var

def minkowski_distance(A, B, p):
    return (sum(abs(a - b) ** p for a, b in zip(A, B))) ** (1 / p)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def knn_predict(X_train, y_train, test_vec, k):
    distances = []
    for i in range(len(X_train)):
        d = minkowski_distance(X_train[i], test_vec, 2)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [label for _, label in neighbors]
    return max(set(labels), key=labels.count)

def confusion_matrix_manual(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def performance_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return accuracy, precision, recall, f1

if __name__ == "__main__":

    csv_path = r"Assignment 3\DCT_mal (2) 1.csv"
    X, y, data = load_dataset(csv_path)
    y_binary = pd.factorize(y)[0]

    A, B = X[0], X[1]
    print("A1 Dot Product:", dot_product(A, B), "| numpy:", np.dot(A, B))
    print("A1 Norm:", euclidean_norm(A), "| numpy:", np.linalg.norm(A))

    mu1, std1 = class_statistics(X, y_binary, 0)
    mu2, std2 = class_statistics(X, y_binary, 1)
    print("A2 Interclass Distance:", interclass_distance(mu1, mu2))

    hist, bins, mean, var = feature_histogram(X[:, 0])
    print("A3 Mean:", mean, "Variance:", var)
    plt.hist(X[:, 0], bins=10)
    plt.title("Histogram")
    plt.show()

    p_vals = range(1, 11)
    distances = [minkowski_distance(A, B, p) for p in p_vals]
    plt.plot(p_vals, distances)
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.title("Minkowski Distance")
    plt.show()

    print("A5 Own Minkowski (p=3):", minkowski_distance(A, B, 3))
    print("A5 SciPy Minkowski (p=3):", minkowski(A, B, 3))

    X_train, X_test, y_train, y_test = split_data(X, y_binary)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print("A8 Accuracy (sklearn kNN):", knn.score(X_test, y_test))

    y_pred = knn.predict(X_test)
    print("A9 First 10 Predictions:", y_pred[:10])

    X_train_sub = X_train[:200]
    y_train_sub = y_train[:200]
    X_test_sub = X_test[:50]
    y_test_sub = y_test[:50]

    y_pred_manual = []
    for x in X_test_sub:
        y_pred_manual.append(knn_predict(X_train_sub, y_train_sub, x, 3))

    y_pred_manual = np.array(y_pred_manual)
    manual_acc = np.mean(y_pred_manual == y_test_sub)
    print("A10 Accuracy (Manual kNN):", manual_acc)

    accs = []
    for k in range(1, 12):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        accs.append(model.score(X_test, y_test))

    print("A11 Accuracies:", accs)
    plt.plot(range(1, 12), accs)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy")
    plt.show()

    TP, TN, FP, FN = confusion_matrix_manual(y_test, y_pred)
    acc, prec, rec, f1 = performance_metrics(TP, TN, FP, FN)

    print("A12 Confusion Matrix")
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

    print("A13 Metrics")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    X_aug = np.c_[np.ones(X_train.shape[0]), X_train]
    w = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y_train
    X_test_aug = np.c_[np.ones(X_test.shape[0]), X_test]
    y_pred_lin = (X_test_aug @ w >= 0.5).astype(int)
    acc_lin = np.mean(y_pred_lin == y_test)
    print("A14 Accuracy (Matrix Inversion):", acc_lin)
