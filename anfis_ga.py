import numpy as np

class AnfisGA:
    def __init__(self, num_inputs, num_rules):
        self.n_inputs = num_inputs
        self.n_rules = num_rules
        
        # DNA Inicial (Pesos Aleatórios)
        self.mu = np.random.randn(num_inputs, num_rules)
        self.sigma = np.ones((num_inputs, num_rules))
        self.consequent_w = np.random.randn(num_rules, num_inputs)
        self.consequent_b = np.random.randn(num_rules)

    def gaussian(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / (sigma + 1e-8))**2)

    def forward(self, x):
        """Calcula a previsão da rede"""
        # 1. Antecedentes
        x_exp = x[:, :, np.newaxis]
        membership = self.gaussian(x_exp, self.mu, self.sigma)
        
        # 2. Regras
        w = np.prod(membership, axis=1)
        w_sum = np.sum(w, axis=1, keepdims=True)
        w_norm = w / (w_sum + 1e-8)
        
        # 3. Consequentes
        rule_output = np.dot(x, self.consequent_w.T) + self.consequent_b
        
        # 4. Saída
        y_pred = np.sum(w_norm * rule_output, axis=1, keepdims=True)
        return y_pred

    # --- FUNÇÕES PARA O ALGORITMO GENÉTICO ---
    def get_chromosome(self):
        """Extrai todos os pesos como uma lista única (Cromossomo)"""
        return np.concatenate([
            self.mu.flatten(),
            self.sigma.flatten(),
            self.consequent_w.flatten(),
            self.consequent_b.flatten()
        ])

    def set_chromosome(self, chromosome):
        """Recebe uma lista genética e reconstrói a rede"""
        idx = 0
        
        # Reconstrói Mu
        size = self.n_inputs * self.n_rules
        self.mu = chromosome[idx : idx+size].reshape(self.n_inputs, self.n_rules)
        idx += size
        
        # Reconstrói Sigma
        self.sigma = chromosome[idx : idx+size].reshape(self.n_inputs, self.n_rules)
        idx += size
        
        # Reconstrói Pesos Lineares
        size = self.n_rules * self.n_inputs
        self.consequent_w = chromosome[idx : idx+size].reshape(self.n_rules, self.n_inputs)
        idx += size
        
        # Reconstrói Bias
        self.consequent_b = chromosome[idx : idx+size]