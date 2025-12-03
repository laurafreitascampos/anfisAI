import numpy as np
from data import get_motor_data
from anfis import AnfisNumpy
from reduce import algoritmo_2_reduce
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
NUM_NODES = 12       # Número de pedaços (Chunks)
EPOCHS = 256         # Épocas de treino
RULES = 8           # Regras
LR = 0.01           # Learning Rate

def split_data_manual(X, y, ratio=0.7):
    # Embaralhar e dividir manualmente
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * ratio)
    
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def run_map_phase():
    # 1. Carregar Dados
    X, y, scaler = get_motor_data()
    
    # 2. Dividir Treino/Teste
    print(f"\n>>> Dividindo dados (70/30)...")
    X_train, y_train, X_test, y_test = split_data_manual(X, y)
    
    # 3. Dividir em Chunks (Nós)
    # np.array_split substitui a lógica complexa de divisão
    X_chunks = np.array_split(X_train, NUM_NODES)
    y_chunks = np.array_split(y_train, NUM_NODES)
    
    models = []
    
    print(f"\n>>> Iniciando Fase MAP ({NUM_NODES} Nós)...")
    
    for i in range(NUM_NODES):
        print(f"\n  [Nó {i+1}] Treinando ANFIS Manual...")
        
        # Criar Rede Manual
        anfis = AnfisNumpy(num_inputs=X.shape[1], num_rules=RULES, learning_rate=LR)
        
        # Loop de Treino Manual
        for ep in range(EPOCHS):
            # 1. Forward: A rede faz a previsão e guarda os cálculos na memória
            pred = anfis.forward(X_chunks[i])
            # 2. Backward: A rede calcula o erro, as derivadas e atualiza os pesos
            # (Retorna o MSE apenas para a gente ver o progresso no print)
            mse = anfis.backward(y_chunks[i])
            
            if (ep + 1) % 50 == 0:
                print(f"    -> Nó {i+1} | Progresso: {ep+1}/{EPOCHS} épocas concluídas.")

        models.append(anfis)
        
        # Teste Rápido (Validação do Nó)
        pred_test = anfis.forward(X_test)
        
        # Desnormalizar
        y_real = scaler.inverse_transform(y_test)
        pred_real = scaler.inverse_transform(pred_test)
        
        rmse = calculate_rmse(y_real, pred_real)
        print(f"    -> Final Nó {i+1}: RMSE = {rmse:.4f}")

    print("\n>>> Fase MAP concluída com sucesso (Rede Manual).")
# --- AQUI ENTRA O ALGORITMO 2 ---
    # Chamamos a função de fusão que acabamos de criar
    super_modelo = algoritmo_2_reduce(models)
    
    # --- TESTE FINAL DO SUPER MODELO ---
    print("\n>>> Testando o Super Modelo Integrado (Ensemble)...")
    
    # Fazer predição com o modelo gigante nos dados de teste
    pred_final = super_modelo.forward(X_test)
    
    # Desnormalizar e calcular erro final
    y_real = scaler.inverse_transform(y_test)
    pred_real_final = scaler.inverse_transform(pred_final)
    
    rmse_final = calculate_rmse(y_real, pred_real_final)
    
    print(f"=== RESULTADO FINAL DO PROJETO ===")
    print(f"RMSE do Ensemble (MapReduce): {rmse_final:.4f}")

    
    # Vamos pegar apenas os primeiros 100 pontos para o gráfico não ficar bagunçado
    amostra = 100
    tempo = np.arange(amostra)
    
    plt.figure(figsize=(12, 6))
    
    # Linha Azul: A Temperatura Real (que foi medida pelo sensor)
    plt.plot(tempo, y_real[:amostra], label='Temperatura Real (°C)', color='blue', linewidth=2)
    
    # Linha Vermelha: O que a sua IA previu
    plt.plot(tempo, pred_real_final[:amostra], label='Previsão da IA (°C)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Validação: Temperatura do Motor (RMSE: {rmse_final:.2f}°C)")
    plt.xlabel("Amostras de Tempo")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()    

    return super_modelo


if __name__ == "__main__":
    # Ajuste na classe para funcionar com a chamada acima
    # Adicionando o helper train_step_full na classe AnfisNumpy se não estiver lá:
    # (Já incluí no código da classe acima)
    run_map_phase()