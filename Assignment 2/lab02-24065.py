# Tushitha-BL.SC.U4AIE24065

import numpy as np
from numpy.linalg import matrix_rank
import pandas as pd
import matplotlib.pyplot as plt

# A1
def calculate_matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)

def calculate_cost(matrix, vector):
    return np.linalg.pinv(matrix)@vector

# A2
def classify_customers(values):
    classifications = []
    for value in values:
        if value > 200:
            classifications.append("RICH")
        else:
            classifications.append("POOR")
    return classifications

# A3
def calculate_mean(values):
    values = list(values)
    length = len(values)
    return sum(values)/length

def calculate_variance(values):
    mean_value = calculate_mean(values)
    values = list(values)
    return sum((val - mean_value)**2 for val in values) / len(values)

def calculate_wednesday_mean(values):
    return np.mean(values)

def calculate_april_mean(values):
    return np.mean(values)

def scatter_plot(x_values, y_values):
    plt.scatter(x_values, y_values)
    plt.xlabel("chg%")
    plt.ylabel("day of the week")
    plt.title("chg% vs day of the week")
    plt.show()

# A4
def explore_thyroid_data(dataset):
    return {
        "data_types": dataset.dtypes,
        "missing_values": dataset.isnull().sum(),
        "mean": dataset.mean(numeric_only=True),
        "variance": dataset.var(numeric_only=True)
    }

# A5
def calculate_frequencies(dataset):
    count_00 = 0
    count_01 = 0
    count_10 = 0
    count_11 = 0
    for i in range(dataset.shape[1]):
        if dataset.iloc[0,i] and dataset.iloc[1,i] == 0:
            count_00+=1
        elif dataset.iloc[0,i]==0 and dataset.iloc[1,i]==1:
            count_01+=1
        elif dataset.iloc[0,i]==1 and dataset.iloc[1,i]==0:
            count_10+=1
        else:
            count_11+=1
    return count_00, count_01, count_10, count_11

def calculate_coefficients(count_00, count_01, count_10, count_11):
    jaccard_coeff = count_11/(count_01+count_10+count_11)
    matching_coeff = (count_00+count_11)/(count_00+count_01+count_10+count_11)
    return jaccard_coeff, matching_coeff

# A6
def convert_to_numeric(dataset):
    for i in range(dataset.shape[0]):
        if dataset.iloc[i,1]=="Graduation" and dataset.iloc[i,2]=="Single":
            dataset.iloc[i,1] = 1
            dataset.iloc[i,2] = 0
    return dataset

def calculate_cosine_similarity(dataset):
    numeric_data = dataset.select_dtypes(include=[np.number])
    return np.dot(numeric_data.iloc[0], numeric_data.iloc[1]) / (np.linalg.norm(numeric_data.iloc[0]) * np.linalg.norm(numeric_data.iloc[1]))

# A7
def generate_similarity_matrix(data_matrix):
    similarity_matrix = np.corrcoef(data_matrix)
    return similarity_matrix

# A8
def impute_missing_values(dataset):
    for column in dataset.columns:
        if dataset[column].dtype != "object":
            dataset[column] = dataset[column].fillna(dataset[column].median())
        else:
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])
    return dataset

# A9
def normalize_numeric_data(dataset):
    numeric_columns = dataset.select_dtypes(include=["int64", "float64"])
    dataset[numeric_columns.columns] = (
        numeric_columns - numeric_columns.min()
    ) / (numeric_columns.max() - numeric_columns.min())
    return dataset

def main():
    purchase_data = pd.read_excel("Lab Session Data.xlsx",sheet_name="Purchase data")
    features = purchase_data.iloc[:,1:4].values
    target = purchase_data.iloc[:,4].values
    
    thyroid_data = pd.read_excel("Lab Session Data.xlsx",sheet_name="thyroid0387_UCI")
    binary_attributes = purchase_data.iloc[:, 5:]

    print("--- A1 RESULTS ---")
    matrix_rank_val = calculate_matrix_rank(features)
    print("The rank of the Matrix A is:", matrix_rank_val)
    
    cost_vector = calculate_cost(features, target)
    print("The calculated cost vector is:\n", cost_vector)

    print("\n--- A2 RESULTS ---")
    customer_classes = classify_customers(target)
    print("The customer classifications (showing first 5) are:", customer_classes[:5])

    print("\n--- A3 RESULTS ---")
    stock_data = pd.read_excel("Lab Session Data.xlsx",sheet_name="IRCTC Stock Price")
    
    mean_val = np.mean(stock_data.iloc[:,3].values)
    variance_val = np.var(stock_data.iloc[:,3].values)
    print(f"The mean of column D is {mean_val} and the variance is {variance_val}")
    
    calculated_mean = calculate_mean(stock_data.iloc[:,3].values)
    print(f"The mean of column D calculated with our own function is: {calculated_mean}")
    
    calculated_variance = calculate_variance(stock_data.iloc[:,3].values)
    print(f"The variance of column D calculated with our own function is: {calculated_variance}")

    wednesday_prices = stock_data[stock_data["Day"]=="Wed"].iloc[:,3].values
    print(f"The mean price on Wednesdays is: {calculate_wednesday_mean(wednesday_prices)}")

    april_prices = stock_data[stock_data["Month"]=="Apr"].iloc[:,3].values
    print(f"The mean price in April is: {calculate_april_mean(april_prices)}")

    loss_condition = lambda x : x<0
    loss_events = loss_condition(stock_data.iloc[:,8].values)
    loss_probability = len(loss_events[loss_events == True])/len(stock_data.iloc[:,8])
    print(f"The probability of making a loss on the stock is: {loss_probability}")

    profit_condition = lambda x: x>0
    profit_events = profit_condition(stock_data[stock_data["Day"]=="Wed"].iloc[:,8].values)
    profit_probability = len(profit_events[profit_events == True])/len(stock_data.iloc[:,8])
    print(f"The probability of checking a profit on Wednesdays is: {profit_probability}")

    probability_is_wednesday = len(stock_data[stock_data["Day"]=="Wed"])/len(stock_data.iloc[:,1])
    print(f"The conditional probability of profit given it is Wednesday is: {profit_probability/probability_is_wednesday}")

    change_percentage = stock_data.iloc[:,8].values
    day_of_week = stock_data.iloc[:,2].values
    print("Generating scatter plot for Change % vs Day of Week...")
    scatter_plot(change_percentage, day_of_week)

    print("\n--- A4 RESULTS ---")
    print("Exploring thyroid data properties:")
    exploration_results = explore_thyroid_data(thyroid_data)
    print(f"Data Types:\n{exploration_results['data_types']}")
    print(f"Missing Values Count:\n{exploration_results['missing_values']}")
    
    print("\n--- A5 RESULTS ---")
    count_00, count_01, count_10, count_11 = calculate_frequencies(binary_attributes.fillna(0)) 
    print(f"Calculated f00: {count_00}, f01: {count_01}, f10: {count_10}, f11: {count_11}")
    
    jaccard_coeff, matching_coeff = calculate_coefficients(count_00, count_01, count_10, count_11)
    print(f"The Jaccard Coefficient is: {jaccard_coeff}")
    print(f"The Simple Matching Coefficient is: {matching_coeff}")

    print("\nA6 RESULTS")
    converted_data = convert_to_numeric(thyroid_data)
    cosine_similarity = calculate_cosine_similarity(converted_data)
    print("Cosine Similarity:", cosine_similarity)
    print("\n--- A7 RESULTS ---")
    similarity_matrix = generate_similarity_matrix(features)
    print("The generated Similarity Matrix is:\n", similarity_matrix)

    print("\n--- A8 RESULTS ---")
    imputed_data = impute_missing_values(thyroid_data.copy())
    print("Missing values have been imputed successfully.")

    print("\n--- A9 RESULTS ---")
    normalized_data = normalize_numeric_data(imputed_data)
    print("Numeric data has been normalized successfully.")

if __name__ == "__main__":
    main()
