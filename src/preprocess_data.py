import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)


    data_clean = data.drop_duplicates()
    # Aplicar One-Hot Encoding
    df_encoded = pd.get_dummies(data_clean, drop_first=True)
    
    # Normalizar la variable numérica 'Age'
    scaler = StandardScaler()
    df_encoded['Age'] = scaler.fit_transform(df_encoded[['Age']])

    X = df_encoded.drop(columns=['Recurred_Yes'])
    y = df_encoded['Recurred_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Reading file paths from command-line arguments
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Save the processed data into separate CSV files
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)
