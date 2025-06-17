from abc import abstractmethod
import numpy as np

class StochModel():

    @abstractmethod
    def __init__(self, num_variables, setting): 
        self.num_variables = num_variables 
        self.expected = setting.get('expectedValue')
        self.devstd = setting.get('devstd')

    
    @abstractmethod
    def simulate_one_time_step(self, parent_node, n_children):

        # Probabilities uniformly distribuited
        prob = 1/n_children*np.ones(n_children)
        
        # Observation simulation made using normal distribution 
        obs = np.random.normal(self.expected, self.devstd, [self.num_variables, n_children]) 
        obs = np.abs(obs)   

        return prob, obs   