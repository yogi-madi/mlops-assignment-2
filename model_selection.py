import pandas as pd
import numpy as np
import os
import joblib
import optuna
import platform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML
from sklearn.ensemble import RandomForestClassifier
# from autosklearn.classification import AutoSklearnClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
    

# Load dataset
df_train = pd.read_csv("feature_engineering/fashion_mnist_train.csv")
df_test = pd.read_csv("feature_engineering/fashion_mnist_test.csv")

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(df_train.iloc[:, :-1], df_train['label'], test_size=0.2, random_state=42)

# Create directory
os.makedirs("model_selection", exist_ok=True)
print("Directory 'model_selection' created or already exists.")

# ==========================
# 1️⃣ AutoML Model Selection
# ==========================
model_results = []

# TPOT
print("Running TPOT AutoML...")
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1)
tpot.fit(X_train, y_train)
y_pred = tpot.fitted_pipeline_.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
model_results.append(["TPOT", accuracy])
joblib.dump(tpot.fitted_pipeline_, "model_selection/tpot_model.pkl")

# H2O.ai
print("Running H2O AutoML...")
h2o.init()
train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
val_h2o = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))
aml = H2OAutoML(max_models=5, seed=42)
aml.train(y=train_h2o.columns[-1], training_frame=train_h2o)
print("H2O AutoML training completed.")
print("Best model:", aml.leader)
# y_pred = aml.leader.predict(val_h2o).as_data_frame().values.flatten()
y_pred = aml.leader.predict(val_h2o).as_data_frame()["predict"].values.astype(int)
accuracy = accuracy_score(y_val, y_pred)
model_results.append(["H2O.ai", accuracy])
print("Saving H2O AutoML model...")
aml.leader.download_mojo(path="model_selection/")#, filename="h2o_automl_model.zip")
print("H2O AutoML model saved as 'model_selection/h2o_automl_model.zip'.")


# Save AutoML results
df_results = pd.DataFrame(model_results, columns=["Model", "Accuracy"])
print("Saving AutoML results...")
df_results.to_csv("model_selection/automl_results.csv", index=False)
print("AutoML results saved in 'model_selection/automl_results.csv'")

# Shutdown H2O server
h2o.shutdown(prompt=False)

# ==============================
# 2️⃣ Hyperparameter Optimization
# ==============================
print("Running Hyperparameter Optimization with Optuna...")

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 50, step=5)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    # Log each trial result
    with open("model_selection/hyperparameter_tuning_log.txt", "a") as log_file:
        log_file.write(f"Trial {trial.number}: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, Accuracy={accuracy}\n")

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Save best parameters
with open("model_selection/best_hyperparameters.txt", "w") as f:
    f.write(str(best_params))

print("Model selection & hyperparameter tuning completed. Check 'model_selection/' for results.")
