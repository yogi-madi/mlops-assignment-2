import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and convert to DataFrame
df_train = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
df_train['label'] = y_train

df_test = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))
df_test['label'] = y_test

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
df_train.iloc[:, :-1] = scaler.fit_transform(df_train.iloc[:, :-1])
df_test.iloc[:, :-1] = scaler.transform(df_test.iloc[:, :-1])

# Save processed data
os.makedirs("feature_engineering", exist_ok=True)
df_train.to_csv("feature_engineering/fashion_mnist_train.csv", index=False)
df_test.to_csv("feature_engineering/fashion_mnist_test.csv", index=False)

print("Feature engineering completed. Normalized dataset saved in 'feature_engineering/' directory.")
