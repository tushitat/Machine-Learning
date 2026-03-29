import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def load_dataset():
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "DCT_mal (2) 1.csv")
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1


def tune_model(model, params, X_train, y_train):
    if params:
        n_iter = min(10, np.prod([len(v) for v in params.values()]))
        search = RandomizedSearchCV(
            model,
            params,
            n_iter=n_iter,
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        return model.fit(X_train, y_train)


if __name__ == "__main__":

    X, y = load_dataset()
    y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "SVM": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 10]}),
        "RandomForest": (RandomForestClassifier(), {"n_estimators": [50, 100]}),
        "AdaBoost": (AdaBoostClassifier(), {"n_estimators": [50, 100]}),
        "NaiveBayes": (GaussianNB(), {}),
        "MLP": (MLPClassifier(max_iter=500), {"hidden_layer_sizes": [(50,), (100,)]})
    }

    results = []

    for name, (model, params) in models.items():

        best_model = tune_model(model, params, X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metrics = evaluate_model(y_train, y_train_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        results.append((name, train_metrics, test_metrics))

    for name, train_m, test_m in results:
        print(name)
        print("Train:", train_m)
        print("Test :", test_m)
        print()