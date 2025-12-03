import sympy as sp

def gerar_formulas_gradiente():
    # Definindo símbolos matemáticos
    x, mu, sigma = sp.symbols('x mu sigma')
    
    # Eq. 1 do Artigo: Função Gaussiana
    # gauss = exp( -0.5 * ((x - mu) / sigma)^2 )
    term = (x - mu) / sigma
    gauss = sp.exp(-0.5 * term**2)
    
    print("--- Calculando Derivadas Simbólicas com SymPy ---")
    
    # 1. Derivada em relação ao Centro (mu)
    d_gauss_d_mu = sp.diff(gauss, mu)
    print(f"\n1. Derivada d(Gauss)/d(mu):\n{d_gauss_d_mu}")
    
    # 2. Derivada em relação à Largura (sigma)
    d_gauss_d_sigma = sp.diff(gauss, sigma)
    print(f"\n2. Derivada d(Gauss)/d(sigma):\n{d_gauss_d_sigma}")
    
    print("\n--- Códigos Python Gerados (Para copiar e colar) ---")
    print("# Para Mu:")
    print(sp.python(d_gauss_d_mu))
    print("\n# Para Sigma:")
    print(sp.python(d_gauss_d_sigma))

if __name__ == "__main__":
    gerar_formulas_gradiente()