import numpy as np
from data.parameters import parameters_ATO
from data.parameters import parameters_NewsVendor

scenario_setting_ATO = {
    'expectedValue' : np.array([400]), # sarebbe meglio impostare un vettore di medie differenti per ogni componente
    'devstd' : 300,
}

scenario_setting_NewsVendor = {
    'expectedValue' : np.array([60]),
    'devstd' : 6,
}