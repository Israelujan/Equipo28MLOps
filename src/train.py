import yaml
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import mlflow
import os


def load_params():
    with open("params.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


# Entrenamiento del modelo
def train_model(X_train, y_train, X_test, y_test, model_type, min_accuracy=0.85):
    model_params = params["models"][model_type]

    if model_type == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params)

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")

        # Verificar que la precisión cumpla con el mínimo requerido
        if acc < min_accuracy:
            raise ValueError(f"Accuracy {acc} is below the minimum threshold of {min_accuracy}")

        # Loggear métricas en MLflow
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})

        # Loggear hiperparámetros
        mlflow.log_params(model_params)

        # Loggear el modelo en MLflow
        mlflow.sklearn.log_model(model, f"{model_type}_model")

        # Registrar el modelo en el Modelo Registry de MLflow
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_type}_model"
        registered_model = mlflow.register_model(model_uri, f"{model_type}_registry")

        # Añadir etiquetas a la versión registrada del modelo en el Model Registry
        mlflow.tracking.MlflowClient().set_model_version_tag(
            name=f"{model_type}_registry",
            version=registered_model.version,
            key="stage",
            value="staging",
        )
        mlflow.tracking.MlflowClient().set_model_version_tag(
            name=f"{model_type}_registry",
            version=registered_model.version,
            key="version",
            value="1.0.0",
        )
        mlflow.tracking.MlflowClient().set_model_version_tag(
            name=f"{model_type}_registry",
            version=registered_model.version,
            key="risk_level",
            value="low",
        )

    return model, acc


if __name__ == "__main__":
    # Rutas por defecto para los datos de entrenamiento y prueba
    default_data_path = "data/raw/Thyroid_Diff.csv"

    # Verificar si se pasaron rutas como argumentos
    if len(sys.argv) == 5:
        X_train_path = sys.argv[1]
        y_train_path = sys.argv[2]
        X_test_path = sys.argv[3]
        y_test_path = sys.argv[4]
    else:
        # Cargar datos directamente desde el archivo por defecto
        data = pd.read_csv(default_data_path)
        
        # Dividir los datos en entrenamiento y prueba
        X = data.drop(columns=["Recurred_Yes"])  # Ajusta el nombre de la columna objetivo según sea necesario
        y = data["Recurred_Yes"]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = "logistic_regression"  # Puedes cambiarlo a "random_forest" si lo prefieres
    model, acc = train_model(X_train, y_train, X_test, y_test, model_type)
    print(f"Modelo entrenado con precisión de {acc:.2f}")
