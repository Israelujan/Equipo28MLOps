import yaml
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import mlflow

def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

params = load_params()
#Entrenamiento del modelo
def train_model(X_train_path, y_train_path, X_test_path, y_test_path, model_type):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    model_params = params['models'][model_type]

    
    if model_type == 'logistic_regression':
        model = LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)

    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')

        # Loggear métricas en MLflow
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})

        # Loggear hiperparámetros
        mlflow.log_params(model_params)
        
        # Loggear el modelo en MLflow
        mlflow.sklearn.log_model(model, f"{model_type}_model")
        
        # Registrar el modelo en el Modelo Registry de MLflow
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_type}_model"
        mlflow.register_model(model_uri, f"{model_type}_registry")

    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    X_test_path = sys.argv[3]
    y_test_path = sys.argv[4]
    model_type = sys.argv[5]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.pkl"

    model = train_model(X_train_path, y_train_path, X_test_path, y_test_path, model_type)
    joblib.dump(model, model_path)
