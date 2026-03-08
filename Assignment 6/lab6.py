import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

def calculate_entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    entropy = 0
    for p in probs:
        entropy += -p * np.log2(p)
    return entropy

def equal_width_binning(data, bins=4):
    min_val = np.min(data)
    max_val = np.max(data)
    width = (max_val - min_val) / bins

    binned = np.floor((data - min_val) / width)
    binned[binned == bins] = bins - 1
    return binned.astype(int)

def equal_frequency_binning(data, bins=4):
    return pd.qcut(data, bins, labels=False, duplicates="drop")

def calculate_gini(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    gini = 1 - np.sum(probs ** 2)

    return gini

def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[:, feature], return_counts=True)
    weighted_entropy = 0

    for v, c in zip(values, counts):
        subset_y = y[X[:, feature] == v]
        weighted_entropy += (c / len(y)) * calculate_entropy(subset_y)
    ig = total_entropy - weighted_entropy

    return ig

def find_best_root_feature(X, y):
    num_features = X.shape[1]
    gains = []
    for f in range(num_features):
        ig = information_gain(X, y, f)
        gains.append(ig)
    best_feature = np.argmax(gains)
    return best_feature, gains[best_feature]

def bin_features(X, bins=4, method="width"):
    X_binned = np.zeros_like(X)
    for i in range(X.shape[1]):
        if method == "width":
            X_binned[:, i] = equal_width_binning(X[:, i], bins)
        elif method == "frequency":
            X_binned[:, i] = equal_frequency_binning(X[:, i], bins)
    return X_binned

def build_decision_tree(X, y):
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=0
    )

    model.fit(X, y)
    return model

def visualize_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Decision Tree for Palm Leaf Manuscript Classification")
    plt.show()

def plot_decision_boundary(X, y):
    X2 = X[:, :2]

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="viridis")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Decision Boundary (Palm Leaf Manuscript Classification)")
    plt.show()

def load_dataset(path):
    data = pd.read_csv(path)
    X = data.drop("LABEL", axis=1).values
    y = data["LABEL"].values
    return X, y, data.columns[:-1]

def main():
    dataset_path = "DCT_mal (2) 1.csv"
    X, y, feature_names = load_dataset(dataset_path)
    print("Dataset Loaded")
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])
    entropy = calculate_entropy(y)
    print("\nDataset Entropy:", entropy)
    gini = calculate_gini(y)
    print("Dataset Gini Index:", gini)
    X_binned = bin_features(X, bins=4, method="width")
    best_feature, gain = find_best_root_feature(X_binned, y)
    print("\nBest Root Feature:", best_feature)
    print("Information Gain:", gain)
    tree_model = build_decision_tree(X_binned, y)
    visualize_tree(tree_model, feature_names)
    plot_decision_boundary(X_binned, y)

if __name__ == "__main__":
    main()