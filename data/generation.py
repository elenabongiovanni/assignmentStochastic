import numpy as np

def generate_random_demand(size, low, high):
    """Genera un vettore di numeri randomici per la domanda"""
    return np.random.randint(low, high + 1, size=size)

def generate_random_prob(size):
    """Genera un vettore di numeri randomici per le probabilità e lo normalizza"""
    random_prob = np.random.rand(size)
    return random_prob / random_prob.sum()


