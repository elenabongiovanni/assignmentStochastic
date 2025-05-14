import numpy as np

class Generation():
    
    def __init__(self, num_var, num_children, seed, a, c, m):
        self.scenarios = np.zeros(num_var,num_children)
        self.means = np.zeros(num_children)
        self.random_generated = random_number_generation(seed, a, c, m, num_children)
        self.mean = mc_mean(self.random_generated)
        self.variance = mc_variance(self.random_generated, self.mean)


def random_number_generation(seed, a, c, m, num_generated):
    generated = np.zeros(num_generated)
    generated[0] = seed

    for i in range(1,num_generated):
        generated[i]  = (a*generated[i-1] + c)%m

    return generated

def mc_mean(generated):
    mean = sum(generated)/len(generated)
    return mean

def mc_variance(generated,mean):
    variance = (sum(generated-mean)^2)/len(generated)^2
    return variance



