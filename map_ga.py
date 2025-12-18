import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count # <--- A Mágica acontece aqui
import time

from data import get_motor_data
from reduce import algoritmo_2_reduce
from genetic_optimizer import GeneticOptimizer

# --- CONFIGURAÇÕES (Mantivemos as mesmas) ---
NUM_NODES = 12
GENERATIONS = 100
POP_SIZE = 30
RULES = 8
MUTATION = 0.05
CROSSOVER = 0.8

# --- FUNÇÃO DE TRABALHO (WORKER) ---
# Esta função precisa ficar fora do 'run' para o multiprocessamento funcionar
def train_single_node(args):
    """
    Função isolada que treina UM único nó.
    Recebe todos os dados empacotados em uma tupla 'args'.
    """
    node_id, X_chunk, y_chunk = args
    
    print(f"  [Processo Paralelo] Iniciando Nó {node_id+1}...")
    
    # Cria o otimizador localmente dentro do processo
    optimizer = GeneticOptimizer(POP_SIZE, MUTATION, CROSSOVER, X_chunk.shape[1], RULES)
    
    # Roda a evolução
    best_ind = None
    best_err = float('inf')
    
    for g in range(GENERATIONS):
        ind, err = optimizer.evolve(X_chunk, y_chunk)
        if (g+1) % 20 == 0: # Print menos frequente para não bagunçar o terminal
            print(f"    -> Nó {node_id+1} | Gen {g+1} | MSE: {err:.5f}")
        
        best_ind = ind
        best_err = err
        
    print(f"  [Concluído] Nó {node_id+1} finalizado. MSE Final: {best_err:.4f}")
    return best_ind

def run():
    start_time = time.time()
    
    # 1. Carregar Dados
    X, y, scaler = get_motor_data()
    if X is None: return

    # Divide em Treino/Teste
    split = int(len(X) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Divide em Chunks
    X_chunks = np.array_split(X_train, NUM_NODES)
    y_chunks = np.array_split(y_train, NUM_NODES)
    
    # --- PREPARAÇÃO PARA O MULTIPROCESSAMENTO ---
    # Empacota os dados para enviar para cada núcleo do processador
    tasks = []
    for i in range(NUM_NODES):
        # Cada tarefa é uma tupla com (ID, Dados X, Dados Y)
        tasks.append((i, X_chunks[i], y_chunks[i]))
    
    # Descobre quantos núcleos seu PC tem
    cores = cpu_count()
    print(f"\n>>> Detectei {cores} núcleos de CPU.")
    print(f">>> Iniciando Treinamento Paralelo em {NUM_NODES} Nós...")
    
    # --- A MÁGICA: POOL DE PROCESSOS ---
    # Cria um 'pool' de trabalhadores e distribui as tarefas
    with Pool(processes=min(cores, NUM_NODES)) as pool:
        # O map joga as tarefas para os núcleos e espera todos terminarem
        # Retorna a lista de melhores modelos na ordem certa
        best_models = pool.map(train_single_node, tasks)

    print(f"\n>>> Todos os nós terminaram em {time.time() - start_time:.2f} segundos!")

    # --- FASE REDUCE ---
    super_model = algoritmo_2_reduce(best_models)
    
    # --- TESTE FINAL ---
    print("\n>>> Testando Ensemble...")
    pred = super_model.forward(X_test)
    
    y_real = scaler.inverse_transform(y_test)
    pred_real = scaler.inverse_transform(pred)
    
    rmse_final = np.sqrt(np.mean((y_real - pred_real)**2))
# ... (código anterior do cálculo do RMSE) ...
    print(f"=== RMSE FINAL: {rmse_final:.4f} °C ===")
    
    # --- GRÁFICO ESTILO ARTIGO (REFINADO) ---
    amostra = 200 
    tempo = np.arange(amostra)
    
    # Configurações de fonte para parecer artigo científico
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    plt.figure(figsize=(10, 6)) # Tamanho similar à proporção do artigo
    
    # 1. Linha Azul (Real) - Mais grossa (linewidth=2) e cor azul padrão sólido
    plt.plot(tempo, y_real[:amostra], color='blue', linestyle='-', 
             linewidth=2.0, label='Actual Values')
    
    # 2. Linha Vermelha (Predição) - Tracejado ajustado (dashes) para imitar o artigo
    # dashes=(5, 2) significa: 5 pontos de tinta, 2 de espaço. Fica mais visível.
    plt.plot(tempo, pred_real[:amostra], color='red', linestyle='--', 
             linewidth=2.0, dashes=(5, 2), label='Predicted Values')
    
    # Títulos e Eixos (Em inglês, igual ao artigo)
    plt.title("Ensemble Model Performance After Genetic Algorithm Optimization", fontsize=14, fontweight='bold')
    plt.xlabel("Data Point Index", fontsize=12)
    plt.ylabel("Actual and Predicted Values", fontsize=12)
    
    # Legenda com caixa e sombra, no canto superior direito
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # Grid cinza claro no fundo
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    
    # Ajusta as margens para nada ficar cortado
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
    
    
    
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.facecolor': 'white' # Garante fundo branco
    })

    # Cria a figura
    plt.figure(figsize=(7, 5)) # Tamanho mais quadrado como na imagem
    
    # Define a amostra de 200 pontos (igual ao eixo X da imagem)
    amostra = 200
    tempo = np.arange(amostra)
    
    # 2. Plota os Valores Reais (Linha Azul Sólida)
    # Na imagem, a linha azul parece ter espessura média (aprox 1.5)
    plt.plot(tempo, y_real[:amostra], 
             color='blue', 
             linestyle='-', 
             linewidth=1.2, 
             label='Actual Values')
    
    # 3. Plota a Predição (Linha Vermelha Tracejada)
    # O tracejado da imagem é bem específico (traços curtos).
    plt.plot(tempo, pred_real[:amostra], 
             color='red', 
             linestyle='--', 
             linewidth=1.2, 
             dashes=(4, 2), # (4px de tinta, 2px de espaço) -> Imita o tracejado da foto
             label='Predicted Values')
    
    # 4. Títulos e Legendas EXATOS da imagem
    plt.title("Ensemble Model Performance After Genetic Algorithm Optimization", fontweight='bold')
    plt.xlabel("Data Point Index")
    plt.ylabel("Actual and Predicted Values")
    
    # Ajusta os limites para ficar igual à imagem (0 a 200 no X)
    plt.xlim(0, 200)
    # O Y na imagem vai de ~10 a ~110. O matplotlib faz automático, mas podemos forçar se quiser:
    # plt.ylim(10, 110) 
    
    # 5. Grid e Legenda
    plt.grid(True, linestyle='-', alpha=0.3, color='lightgray') # Grid bem suave
    plt.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False) # Caixa da legenda com borda preta quadrada
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
        run()