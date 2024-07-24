import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("winedata.csv", sep=";")

# Separate features (X) and target variable (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize features (important for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set the MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Model 1: Decision Tree
with mlflow.start_run(run_name="Decision Tree"):
    mlflow.log_param("model_type", "Decision Tree")
    model = DecisionTreeClassifier(random_state=42)
    mlflow.log_param("random_state", "42")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Log Model
    mlflow.sklearn.log_model(model, "DecisionTree")

# Model 2: Logistic Rgression
with mlflow.start_run(run_name="Logistic Regression"):
    mlflow.log_param("model_type", "Logistic Regression")
    model = LogisticRegression(random_state=42)
    mlflow.log_param("random_state", "42")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Log Model
    mlflow.sklearn.log_model(model, "LogisticRegression")

# Model 3: Support Vector Machine
with mlflow.start_run(run_name="Support Vector Machine"):
    mlflow.log_param("model_type", "Support Vector Machine")
    model = SVC(random_state=42)
    mlflow.log_param("random_state", "42")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Log Model
    mlflow.sklearn.log_model(model, "SupportVectorMachine")
