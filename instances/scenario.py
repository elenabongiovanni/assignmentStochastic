import math

class Scenario():

    def __init__(self, demand, prob):
        self.demand = demand
        self.prob = prob
        self.gain = 0

    def add_gain(self, gain):
        self.gain = gain