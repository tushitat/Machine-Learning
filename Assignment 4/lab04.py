import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y, data

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    return cm, precision, recall, f1


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def generate_training_data():
    X = np.random.uniform(1, 10, (20, 2))
    y = np.array([0 if x[0] + x[1] < 10 else 1 for x in X])
    return X, y

def generate_test_grid():
    gx, gy = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
    X_test = np.c_[gx.ravel(), gy.ravel()]
    return X_test, gx, gy

if __name__ == "__main__":

    csv_path = r"Assignment 4\DCT_mal (2) 1.csv"
    X, y, data = load_dataset(csv_path)

    y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    cm_train, p_tr, r_tr, f1_tr = classification_metrics(y_train, y_train_pred)
    cm_test, p_te, r_te, f1_te = classification_metrics(y_test, y_test_pred)

    print("A1 TRAIN Confusion Matrix:\n", cm_train)
    print("Train Precision:", p_tr, "Recall:", r_tr, "F1:", f1_tr)

    print("\nA1 TEST Confusion Matrix:\n", cm_test)
    print("Test Precision:", p_te, "Recall:", r_te, "F1:", f1_te)

    mse, rmse, mape, r2 = regression_metrics(y_test, y_test_pred)
    print("\nA2 Regression Metrics")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R2:", r2)

    X_syn, y_syn = generate_training_data()

    plt.scatter(X_syn[y_syn == 0][:, 0], X_syn[y_syn == 0][:, 1], color="blue")
    plt.scatter(X_syn[y_syn == 1][:, 0], X_syn[y_syn == 1][:, 1], color="red")
    plt.title("Training Data")
    plt.show()

    X_grid, gx, gy = generate_test_grid()

    knn_syn = KNeighborsClassifier(n_neighbors=3)
    knn_syn.fit(X_syn, y_syn)

    y_grid_pred = knn_syn.predict(X_grid)

    plt.scatter(X_grid[y_grid_pred == 0][:, 0], X_grid[y_grid_pred == 0][:, 1], color="blue", s=1)
    plt.scatter(X_grid[y_grid_pred == 1][:, 0], X_grid[y_grid_pred == 1][:, 1], color="red", s=1)
    plt.title("Decision Boundary (k=3)")
    plt.show()

    for k in [1, 3, 5, 7]:
        knn_syn = KNeighborsClassifier(n_neighbors=k)
        knn_syn.fit(X_syn, y_syn)
        y_grid_pred = knn_syn.predict(X_grid)

        plt.scatter(X_grid[y_grid_pred == 0][:, 0], X_grid[y_grid_pred == 0][:, 1], color="blue", s=1)
        plt.scatter(X_grid[y_grid_pred == 1][:, 0], X_grid[y_grid_pred == 1][:, 1], color="red", s=1)
        plt.title(f"Decision Boundary (k={k})")
        plt.show()

    X_proj = X[:, :2]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_proj, y, test_size=0.3, random_state=42, stratify=y
    )

    knn_proj = KNeighborsClassifier(n_neighbors=3)
    knn_proj.fit(Xp_train, yp_train)

    yp_pred = knn_proj.predict(Xp_test)
    print("\nA6 Project Data Accuracy:", knn_proj.score(Xp_test, yp_test))

    param_grid = {"n_neighbors": list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("\nA7 Best k:", grid.best_params_["n_neighbors"])
    print("A7 Best CV Score:", grid.best_score_)
