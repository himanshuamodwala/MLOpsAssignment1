import optuna
import sqlite3
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Optuna objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3, scoring='neg_mean_squared_error').mean()
    return score

# Create an Optuna study
study = optuna.create_study(direction='minimize', storage='sqlite:///optuna_study.db', study_name='rf', load_if_exists=True)
# Optimize the study
study.optimize(objective, n_trials=100)

# Print the best parameters
print("Best parameters: ", study.best_params)

# Train the best model on the entire training dataset
best_model = RandomForestRegressor(**study.best_params)
best_model.fit(X_train, y_train)

# Save the Scaler and Model to a file
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_model, 'best_model.pkl')

# TEST
import sys
