import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load processed data
df_train = pd.read_csv("feature_engineering/fashion_mnist_train.csv")
df_test = pd.read_csv("feature_engineering/fashion_mnist_test.csv")

# Separate features and labels
X_train = df_train.iloc[:, :-1]
y_train = df_train['label']

# Use a sample of data for SHAP (SHAP is computationally expensive)
X_sample = X_train.sample(500, random_state=42)

# Train a simple model (e.g., Decision Tree for interpretation)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_sample, y_train.loc[X_sample.index])

# Explainability with SHAP
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# Generate SHAP summary plot
os.makedirs("explainability", exist_ok=True)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False, feature_names=X_sample.columns)
plt.savefig("explainability/shap_summary.png")
plt.close()

print("Explainability analysis completed. SHAP summary plot saved in 'explainability/' directory.")
