import numpy as np

train_data=np.load('Assignment1-Dataset/train_data.npy')
train_label=np.load('Assignment1-Dataset/train_label.npy')
test_data=np.load('Assignment1-Dataset/test_data.npy')
test_label=np.load('Assignment1-Dataset/test_label.npy')

# Visualize the data
import matplotlib.pyplot as plt

# use panda to show the data
import pandas as pd

# Convert the numpy array to a pandas DataFrame
df_train = pd.DataFrame(train_data)

# Display basic information about the training data
print("Training Data Information:")
print(f"Shape: {train_data.shape}")
print("\nBasic Statistics:")
print(df_train.describe())

# Display the first few rows of the training data
print("\nFirst 5 rows of training data:")
print(df_train.head())



