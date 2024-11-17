import unittest
import pandas as pd
from sklearn.datasets import make_classification
from train import train_model


class TestTrainModel(unittest.TestCase):
    def setUp(self):
        # Crear un dataset simulado para pruebas
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X_train = pd.DataFrame(X[:80])
        self.y_train = pd.Series(y[:80])
        self.X_test = pd.DataFrame(X[80:])
        self.y_test = pd.Series(y[80:])

    def test_model_accuracy(self):
        # Prueba para asegurar que el modelo cumple con una precisión mínima del 85%
        model, accuracy = train_model(self.X_train, self.y_train, self.X_test, self.y_test, model_type="logistic_regression", min_accuracy=0.85)
        self.assertGreaterEqual(accuracy, 0.85, "La precisión del modelo es menor al umbral requerido")

    def test_model_prediction_shape(self):
        # Prueba para asegurar que el modelo produce la cantidad correcta de predicciones
        model, _ = train_model(self.X_train, self.y_train, self.X_test, self.y_test, model_type="logistic_regression")
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape[0], self.X_test.shape[0], "La cantidad de predicciones no coincide con la cantidad de ejemplos de prueba")

if __name__ == '__main__':
    unittest.main()
