import numpy as np
from anfis import AnfisNumpy

def algoritmo_2_reduce(models_list):
    """
    Implementa a Fase REDUCE do Artigo.
    Funde os parâmetros de N modelos em um único Modelo Integrado.
    """
    print("\n>>> Iniciando Algoritmo 2 (Reduce)...")
    
    # 1. Pegar informações básicas do primeiro modelo
    # (Assumimos que todos têm a mesma estrutura de inputs)
    ref_model = models_list[0]
    num_inputs = ref_model.n_inputs
    rules_per_model = ref_model.n_rules
    num_models = len(models_list)
    
    # 2. Calcular o tamanho do Novo Modelo Gigante
    # O artigo diz que o modelo integrado contém a informação de todos os submodelos.
    # Logo, ele terá a SOMA das regras de todos.
    total_rules = rules_per_model * num_models
    
    print(f"    Fundindo {num_models} modelos de {rules_per_model} regras...")
    print(f"    Novo 'Super Modelo' terá {total_rules} regras.")
    
    # 3. Criar o objeto do Super Modelo (Vazio)
    super_model = AnfisNumpy(num_inputs, total_rules)
    
    # 4. A FUSÃO (Concatenar Parâmetros)
    # Aqui usamos Numpy para "colar" as matrizes umas nas outras
    
    # -- Antecedentes (Gaussianas) --
    # mu shape original: (Inputs, Rules) -> Concatenamos no eixo 1 (colunas/regras)
    super_model.mu = np.concatenate([m.mu for m in models_list], axis=1)
    super_model.sigma = np.concatenate([m.sigma for m in models_list], axis=1)
    
    # -- Consequentes (Lineares) --
    # weights shape original: (Rules, Inputs) -> Concatenamos no eixo 0 (linhas/regras)
    super_model.consequent_w = np.concatenate([m.consequent_w for m in models_list], axis=0)
    super_model.consequent_b = np.concatenate([m.consequent_b for m in models_list], axis=0)
    
    print("    Fusão concluída com sucesso!")
    return super_model