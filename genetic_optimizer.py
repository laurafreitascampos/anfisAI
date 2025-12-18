import numpy as np
import copy
from anfis_ga import AnfisGA

class GeneticOptimizer:
    def __init__(self, pop_size, mutation_rate, crossover_rate, num_inputs, num_rules):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        
        # Cria a população inicial (Vários modelos AnfisGA)
        self.population = [AnfisGA(num_inputs, num_rules) for _ in range(pop_size)]

    def calculate_fitness(self, X, y):
        fitness_scores = []
        errors = []
        
        for individual in self.population:
            pred = individual.forward(X)
            mse = np.mean((y - pred)**2)
            errors.append(mse)
            # Fitness é o inverso do erro (Menor erro = Maior nota)
            fitness_scores.append(1.0 / (mse + 1e-8))
            
        return np.array(fitness_scores), np.array(errors)

    def selection_roulette(self, fitness_scores):
        """Roleta Viciada: Quem tem mais fitness tem mais chance"""
        total_fitness = np.sum(fitness_scores)
        probs = fitness_scores / total_fitness
        selected_indices = np.random.choice(range(self.pop_size), size=self.pop_size, p=probs)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """Crossover de 1 Ponto (conforme exercício)"""
        if np.random.rand() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        dna1 = parent1.get_chromosome()
        dna2 = parent2.get_chromosome()
        
        # Escolhe um ponto de corte aleatório no DNA
        cut = np.random.randint(1, len(dna1)-1)
        
        # Cruza
        child1_dna = np.concatenate([dna1[:cut], dna2[cut:]])
        child2_dna = np.concatenate([dna2[:cut], dna1[cut:]])
        
        c1 = AnfisGA(self.num_inputs, self.num_rules)
        c1.set_chromosome(child1_dna)
        c2 = AnfisGA(self.num_inputs, self.num_rules)
        c2.set_chromosome(child2_dna)
        
        return c1, c2

    def mutation(self, individual):
        """Mutação: Adiciona ruído em alguns genes"""
        dna = individual.get_chromosome()
        mask = np.random.rand(len(dna)) < self.mutation_rate
        noise = np.random.randn(len(dna)) * 0.02
        dna[mask] += noise[mask]
        individual.set_chromosome(dna)
        return individual

    def evolve(self, X, y):
        # 1. Avalia todo mundo
        fitness, errors = self.calculate_fitness(X, y)
        best_idx = np.argmax(fitness)
        best_ind = copy.deepcopy(self.population[best_idx])
        best_err = errors[best_idx]
        
        # 2. Seleção (Roleta)
        parents = self.selection_roulette(fitness)
        
        # 3. Cruzamento e Mutação
        new_pop = []
        for i in range(0, self.pop_size, 2):
            p1, p2 = parents[i], parents[(i+1)%self.pop_size]
            c1, c2 = self.crossover(p1, p2)
            new_pop.extend([self.mutation(c1), self.mutation(c2)])
            
        # 4. Elitismo (Mantém o campeão vivo)
        self.population = new_pop[:self.pop_size]
        self.population[0] = best_ind
        
        return best_ind, best_err