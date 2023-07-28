# Importing the necessary libraries\
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
data = pd.read_csv('C:\Repos Personales\MachineLearningAz\External Datasets\diabetes.csv')

# Identify missing data (assumes that missing data is represented as NaN)
print(data.dtypes)

# Print the number of missing entries in each column
print(data.isnull().sum())

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# Fit the imputer on the DataFrame
imputer.fit(data)

# Apply the transform to the DataFrame
data = imputer.transform(data)

#Print your updated matrix of features
print(data)