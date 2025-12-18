import numpy as np
from anfis_ga import AnfisGA # <--- Mudança aqui

def algoritmo_2_reduce(models_list):
    print("\n>>> [Reduce] Fundindo modelos...")
    
    ref = models_list[0]
    total_rules = ref.n_rules * len(models_list)
    
    super_model = AnfisGA(ref.n_inputs, total_rules) # <--- Mudança aqui
    
    # Cola as matrizes
    super_model.mu = np.concatenate([m.mu for m in models_list], axis=1)
    super_model.sigma = np.concatenate([m.sigma for m in models_list], axis=1)
    super_model.consequent_w = np.concatenate([m.consequent_w for m in models_list], axis=0)
    super_model.consequent_b = np.concatenate([m.consequent_b for m in models_list], axis=0)
    
    print(f"    Novo Super Modelo tem {total_rules} regras.")
    return super_model