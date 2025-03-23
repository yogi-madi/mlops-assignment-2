import pandas as pd
import tensorflow as tf
# from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
import os
# from pydantic_settings import BaseSettings

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Convert data to DataFrame
df_train = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
df_train['label'] = y_train

df_test = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))
df_test['label'] = y_test

# Generate EDA Report
profile = ProfileReport(df_train, title="Fashion MNIST EDA Report", explorative=True, minimal=True)

# Save report
os.makedirs("eda", exist_ok=True)
profile.to_file("eda/fashion_mnist_eda.html")

print("EDA report saved successfully in the 'eda/' directory.")