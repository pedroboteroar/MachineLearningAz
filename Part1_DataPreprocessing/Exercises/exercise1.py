# Importing the necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
data = load_iris()

# Identifying the features (independent variables) and the dependent variable
print(data.data, data.target)

# Create dataframe
df_iris = pd.DataFrame(data.data, columns=data.feature_names)

# Create matrix of features (X) and dependent variable vector (y)
X = df_iris.iloc[:, :].values
y = data.target

# Print your feature matrix (X) and dependent variable vector (y)
print(X)
print(y)