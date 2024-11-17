import yaml
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import mlflow
import os


def load_params():
    with open("params.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


# Entrenamiento del modelo
def train_model(X_train_path, y_train_path, model_type):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    model_params = params['models'][model_type]

    if model_type == "logistic_regression":
        model = LogisticRegression(**model_params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**model_params)

    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    # Leer argumentos de línea de comandos
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.pkl"

    # Entrenar el modelo
    print("Iniciando el entrenamiento del modelo...")
    model = train_model(X_train_path, y_train_path, model_type)
    
    # Verificar la instancia del modelo
    print(f"Tipo de modelo entrenado: {type(model)}")
    
    # Verificar si el modelo tiene el método predict
    if hasattr(model, "predict"):
        print("El modelo tiene el método 'predict'.")
    else:
        print("Advertencia: El modelo no tiene el método 'predict'.")
    
    # Verificar atributos comunes del modelo
    if hasattr(model, "coef_"):
        print(f"Coeficientes del modelo (solo para modelos lineales): {model.coef_}")
    if hasattr(model, "intercept_"):
        print(f"Intercepto del modelo: {model.intercept_}")
    if hasattr(model, "classes_"):
        print(f"Clases del modelo: {model.classes_}")

    # Guardar el modelo con pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print("Modelo guardado correctamente.")
