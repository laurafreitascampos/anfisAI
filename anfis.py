import numpy as np

class AnfisNumpy:
    def __init__(self, num_inputs, num_rules, learning_rate=0.01):
        self.n_inputs = num_inputs
        self.n_rules = num_rules
        self.lr = learning_rate
        
        # --- Inicialização de Parâmetros (Random) ---
        # Camada 1: Gaussianas (Mu, Sigma)
        self.mu = np.random.randn(num_inputs, num_rules)
        self.sigma = np.ones((num_inputs, num_rules))
        
        # Camada 4: Lineares (Pesos, Bias)
        self.consequent_w = np.random.randn(num_rules, num_inputs)
        self.consequent_b = np.random.randn(num_rules)

    def gaussian(self, x, mu, sigma):
        # Eq. 1 do artigo: exp(-0.5 * ((x-mu)/sigma)^2)
        return np.exp(-0.5 * ((x - mu) / (sigma + 1e-8))**2)

    def forward(self, x):
        """Passada para Frente (Predição)"""
        self.x_in = x # Cache para o backward
        
        # 1. Fuzzificação (Gaussianas)
        # x: (N, D) -> x_exp: (N, D, 1) para broadcast com (D, R)
        self.x_exp = x[:, :, np.newaxis]
        self.membership = self.gaussian(self.x_exp, self.mu, self.sigma)
        # membership shape: (N, Inputs, Rules)
        
        # 2. Regras (Produto)
        self.w = np.prod(self.membership, axis=1) # (N, Rules)
        
        # 3. Normalização
        self.w_sum = np.sum(self.w, axis=1, keepdims=True)
        self.w_norm = self.w / (self.w_sum + 1e-8) # (N, Rules)
        
        # 4. Consequentes (Linear: Output = x*weights + bias)
        # (N, D) dot (R, D).T -> (N, R)
        self.rule_output = np.dot(x, self.consequent_w.T) + self.consequent_b
        
        # 5. Agregação (Soma Ponderada)
        # Eq. 6 do artigo
        self.y_pred = np.sum(self.w_norm * self.rule_output, axis=1, keepdims=True)
        return self.y_pred

    def backward(self, y_true):
        """Passada para Trás (Treinamento Manual com Fórmulas do SymPy)"""
        N = len(y_true)
        error = self.y_pred - y_true # (N, 1)
        
        # --- A. Gradiente dos Consequentes (Lineares) ---
        # dE/dOut * dOut/dLinear = error * w_norm
        d_linear = error * self.w_norm # (N, Rules)
        
        # Média dos gradientes para o lote
        grad_c_w = np.dot(d_linear.T, self.x_in) / N
        grad_c_b = np.mean(d_linear, axis=0)
        
        # --- B. Gradiente dos Antecedentes (Gaussianas) ---
        # 1. Gradiente em relação a w_norm (normalizado)
        # dE/dWnorm = error * linear_output
        d_wnorm = error * self.rule_output # (N, Rules)
        
        # 2. Gradiente em relação a w (não normalizado - Regra do Quociente)
        # dWnorm/dW = (Sum - w) / Sum^2  (Simplificado)
        # Termo comum: (d_wnorm - média_ponderada_erro) / w_sum
        sum_d_wnorm_w = np.sum(d_wnorm * self.w, axis=1, keepdims=True)
        d_w = (d_wnorm * self.w_sum - sum_d_wnorm_w) / (self.w_sum**2 + 1e-8) # (N, Rules)
        
        # 3. Gradiente em relação à Membership (Regra do Produto)
        # w = prod(mu) -> dw/dmu = w / mu
        # Expandir d_w para (N, 1, Rules)
        d_w_exp = d_w[:, np.newaxis, :]
        d_membership = d_w_exp * (self.w[:, np.newaxis, :] / (self.membership + 1e-8))
        
        # --- AQUI ENTRAM AS FÓRMULAS DO SYMPY ---
        # O SymPy nos disse que: dGauss/dMu = Gauss * (x - mu) / sigma^2
        # E dGauss/dSigma = Gauss * (x - mu)^2 / sigma^3
        
        diff = self.x_exp - self.mu
        
        # Aplicação Vetorizada das fórmulas do SymPy:
        d_mu_local = d_membership * self.membership * (diff / (self.sigma**2 + 1e-8))
        d_sigma_local = d_membership * self.membership * ((diff**2) / (self.sigma**3 + 1e-8))
        
        # Média sobre o lote
        grad_mu = np.mean(d_mu_local, axis=0)
        grad_sigma = np.mean(d_sigma_local, axis=0)
        
        # --- C. Atualização (Gradient Descent) ---
        self.consequent_w -= self.lr * grad_c_w
        self.consequent_b -= self.lr * grad_c_b
        self.mu           -= self.lr * grad_mu
        self.sigma        -= self.lr * grad_sigma
        
        return np.mean(error**2) # Retorna MSE