# Import required libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import lime
import lime.lime_tabular


# ==============================
# Function 1: Load Dataset
# ==============================
def load_data():
    # Example dataset (replace with your project dataset)
    from sklearn.datasets import load_breast_cancer
    cancer_dataset = load_breast_cancer()

    features = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
    target_labels = pd.Series(cancer_dataset.target)

    return features, target_labels


# ==============================
# Function 2: Split Data
# ==============================
def split_data(features, target_labels):
    return train_test_split(features, target_labels, test_size=0.2, random_state=42)


# ==============================
# Function 3: Create Base Models
# ==============================
def get_base_models():
    base_models = [
        ('decision_tree', DecisionTreeClassifier()),
        ('random_forest', RandomForestClassifier(n_estimators=100)),
        ('gradient_boosting', GradientBoostingClassifier())
    ]
    return base_models


# ==============================
# Function 4: Build Stacking Model
# ==============================
def build_stacking_model(base_models):
    meta_classifier = LogisticRegression()

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_classifier
    )

    return stacking_model


# ==============================
# Function 5: Build Pipeline
# ==============================
def build_pipeline(model):
    machine_learning_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    return machine_learning_pipeline


# ==============================
# Function 6: Train Model
# ==============================
def train_model(machine_learning_pipeline, features_train, labels_train):
    machine_learning_pipeline.fit(features_train, labels_train)
    return machine_learning_pipeline


# ==============================
# Function 7: Evaluate Model
# ==============================
def evaluate_model(machine_learning_pipeline, features_test, labels_test):
    model_predictions = machine_learning_pipeline.predict(features_test)
    model_accuracy = accuracy_score(labels_test, model_predictions)
    return model_accuracy, model_predictions


# ==============================
# Function 8: LIME Explanation
# ==============================
def explain_with_lime(machine_learning_pipeline, features_train, features_test):
    feature_names = features_train.columns.tolist()

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=features_train.values,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode='classification'
    )

    def predict_wrapper(data_array):
        data_dataframe = pd.DataFrame(data_array, columns=feature_names)
        return machine_learning_pipeline.predict_proba(data_dataframe)

    lime_explanation = lime_explainer.explain_instance(
        data_row=features_test.iloc[0].values,
        predict_fn=predict_wrapper
    )

    return lime_explanation


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    # Load and split data
    features, target_labels = load_data()
    features_train, features_test, labels_train, labels_test = split_data(features, target_labels)

    # Build models
    base_models = get_base_models()
    stacking_model = build_stacking_model(base_models)

    # Build pipeline
    machine_learning_pipeline = build_pipeline(stacking_model)

    # Train
    trained_pipeline = train_model(machine_learning_pipeline, features_train, labels_train)

    # Evaluate
    model_accuracy, sample_predictions = evaluate_model(trained_pipeline, features_test, labels_test)

    # LIME explanation
    lime_explanation = explain_with_lime(trained_pipeline, features_train, features_test)

    # Print results (ONLY here)
    print("Model Accuracy:", model_accuracy)
    print("\nSample Predictions:", sample_predictions[:5])

    print("\nLIME Explanation:")
    for feature_name, feature_weight in lime_explanation.as_list():
        print(f"{feature_name}: {feature_weight}")


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()