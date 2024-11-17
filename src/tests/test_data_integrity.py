import unittest
import pandas as pd

class TestDataIntegrity(unittest.TestCase):
    def setUp(self):
        # Cargar el dataset
        self.df = pd.read_csv("data/raw/Thyroid_Diff.csv")

    def test_no_missing_values(self):
        # Verificar que no haya valores nulos
        self.assertFalse(self.df.isnull().any().any(), "El dataset contiene valores nulos.")

    def test_column_types(self):
        # Verificar los tipos de datos esperados (ajustar según el dataset)
        expected_types = {
            'Age': 'int64',
            'Gender': 'object',
            # Agrega otras columnas con los nombres y tipos de datos correctos
            # Asegúrate de revisar los nombres exactos en el dataset
        }
        for column, expected_type in expected_types.items():
            with self.subTest(column=column):
                if column in self.df.columns:
                    actual_type = str(self.df[column].dtype)
                    self.assertEqual(actual_type, expected_type, f"El tipo de dato de {column} es {actual_type} pero se esperaba {expected_type}.")
                else:
                    self.fail(f"La columna {column} no está presente en el dataset.")

    def test_unique_values_in_categorical(self):
        # Comprobar valores únicos en las columnas categóricas
        categorical_columns = ['Gender']  # Ajusta según las columnas categóricas en el dataset
        max_unique_values = 10  # Umbral de valores únicos razonable

        for column in categorical_columns:
            if column in self.df.columns:
                unique_values = self.df[column].nunique()
                with self.subTest(column=column):
                    self.assertLessEqual(unique_values, max_unique_values, f"La columna {column} tiene demasiados valores únicos.")
            else:
                self.fail(f"La columna {column} no está presente en el dataset.")

    def test_value_ranges(self):
        # Verificar que ciertos valores estén dentro de rangos lógicos
        if 'Age' in self.df.columns:
            self.assertTrue(self.df['Age'].between(0, 120).all(), "La columna Age tiene valores fuera de rango.")
        if 'Outcome' in self.df.columns:
            self.assertTrue(self.df['Outcome'].isin([0, 1]).all(), "La columna Outcome contiene valores fuera de 0 y 1.")

    def test_no_duplicates(self):
        # Verificar que no haya filas duplicadas
        duplicates = self.df.duplicated().sum()
        self.assertEqual(duplicates, 0, f"El dataset contiene {duplicates} filas duplicadas.")

if __name__ == '__main__':
    unittest.main()