import pandas as pd
import numpy as np

class ManualScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit_transform(self, data):
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        numerator = data - self.min_val
        denominator = (self.max_val - self.min_val) + 1e-8
        return numerator / denominator

    def inverse_transform(self, data_norm):
        if self.min_val is None:
            raise Exception("O Scaler ainda não foi treinado!")
        return data_norm * ((self.max_val - self.min_val) + 1e-8) + self.min_val

def get_motor_data():
    print(">>> 1. Lendo arquivo CSV...")
    # ATENÇÃO: Verifique se o nome do arquivo bate com o que você tem na pasta
    filename = "measures_v2.csv" 
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"ERRO: Não achei o arquivo {filename} na pasta!")
        return None, None, None

    features = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q']
    target = 'stator_tooth'
    
    X = df[features].values
    y = df[target].values.reshape(-1, 1)
    
    print(">>> 2. Normalizando dados...")
    scaler_x = ManualScaler()
    scaler_y = ManualScaler()
    
    X_norm = scaler_x.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)
    
    return X_norm, y_norm, scaler_y