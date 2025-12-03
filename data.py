import pandas as pd
import numpy as np

class ManualScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit_transform(self, data):
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        # Adiciona 1e-8 para evitar divisão por zero
        numerator = data - self.min_val
        denominator = (self.max_val - self.min_val) + 1e-8
        return numerator / denominator

    def inverse_transform(self, data_norm):
        if self.min_val is None:
            raise Exception("O Scaler ainda não foi treinado!")
        return data_norm * ((self.max_val - self.min_val) + 1e-8) + self.min_val

# --- FUNÇÃO PRINCIPAL ---
def get_motor_data():
    print(">>> 1. Verificando arquivo local...")
    
    # --- MUDANÇA AQUI: Nome exato do seu arquivo ---
    filename = "measures_v2.csv"
    
    print(">>> 2. Lendo arquivo CSV...")
    df = pd.read_csv(filename)
    
    # 3. Filtrar Colunas (Conforme Artigo)
    print(f"Colunas encontradas no arquivo: {list(df.columns)}")
    # Verifique se o measures_v2.csv tem essas colunas exatas
    features = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q']
    target = 'stator_tooth'
    
    X = df[features].values
    y = df[target].values.reshape(-1, 1)
    
    # 4. Normalizar
    print(">>> 3. Normalizando dados manualmente (Algoritmo 1)...")
    scaler_x = ManualScaler()
    scaler_y = ManualScaler()
    
    X_norm = scaler_x.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)
    
    print(f"    Sucesso! Matriz X: {X_norm.shape}, Vetor y: {y_norm.shape}")
    return X_norm, y_norm, scaler_y

if __name__ == "__main__":
    get_motor_data()