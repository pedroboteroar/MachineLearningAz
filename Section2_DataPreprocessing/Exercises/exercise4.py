# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
df_iris = load_iris()

# Separate features and target
X = df_iris.data
y = df_iris.target

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Print the dataset split
print(X_train, X_test, y_train, y_test)

# Apply feature scaling on the training and test sets
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Print the scaled training and test sets
print(X_train, X_test)