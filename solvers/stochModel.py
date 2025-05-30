from abc import abstractmethod
import numpy as np

class StochModel():

    @abstractmethod
    def __init__(self, num_variables, setting): 
        self.num_variables = num_variables  # Numero di variabili per scenario (ad esempio, numero di prodotti)
        self.expected = setting.get('expectedValue')
        self.devstd = setting.get('devstd')

    
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
        - obs: Matrice di osservazioni, una riga per variabile, ogni colonna per scenario
        """

        # Simulazione delle probabilità (uniformi o distribuite secondo un altro criterio)
        prob = 1/n_children*np.ones(n_children)
        
        # Simulazione delle osservazioni: matrice con num_variables righe e n_children colonne
        # Ogni colonna rappresenta una variabile stocastica per scenario
        obs = np.random.normal(self.expected, self.devstd, [self.num_variables, n_children]) 
        obs = np.abs(obs)   # Per evitare valori negativi (puoi rimuoverlo se non necessario)
        # self.expected = sum(prob[i] * obs[:,i] for i in range(n_children)) 

        return prob, obs   