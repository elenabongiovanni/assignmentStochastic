# -*- coding: utf-8 -*-
from abc import abstractmethod
import numpy as np

class StochModel():
    @abstractmethod
    
    def __init__(self, num_variables):
        self.num_variables = num_variables  # Numero di variabili per scenario (ad esempio, numero di prodotti)
        self.expected = 80

    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children):
        """
        Simula uno step temporale stocastico.
        
        parent_node: Nodo da cui partire per generare i figli. Questo nodo può contenere
                      variabili come la domanda iniziale, ma non è direttamente usato
                      per la simulazione in questa versione.
        
        n_children: Numero di figli (scenari) da generare. Ogni figlio avrà un insieme
                    di osservazioni (una riga della matrice).
        
        Restituisce:
        - prob: Vettore di probabilità per ciascun scenario
        - obs: Matrice di osservazioni, una riga per scenario, ogni colonna per variabile
        """
        # Simulazione delle probabilità (uniformi o distribuite secondo un altro criterio)
        prob = np.random.rand(n_children)
        prob /= prob.sum()  # Normalizzazione per ottenere probabilità valide
        
        # Simulazione delle osservazioni: matrice con n_children righe e num_variables colonne
        # Ogni colonna rappresenta una variabile stocastica per scenario
        obs = np.random.randn(self.num_variables,n_children)  # Ad esempio, variabili normali
        obs = np.abs(obs)  # Per evitare valori negativi (puoi rimuoverlo se non necessario)
        self.expected = sum(prob[i] * obs[i:1] for i in range(n_children)) 
        return prob, obs

        # parent_node is the node that the generate the two-stage subtree that
        # we are going to build and add to the general scenario tree
        #pass