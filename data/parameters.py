import numpy as np

# Parametri ATO
costs = np.array([6,5,10,20,3])
selling_prices = np.array([60,20,20])
machine_capacities =np.array([800,700,600])

parameters_ATO = {
    'n_components' : len(costs),
    'n_items' : len(selling_prices),
    'costs' : costs,
    'selling_prices' : selling_prices,
    'n_machines' : len(machine_capacities),
    'machine_capacities' : machine_capacities,
    'gozinto_factor' : np.array([[1, 1, 0], [0,1,1], [1,0,0], [0,1,0], [0,0,1]]),
    'time_required' : np.array([[1,2,1], [1,2,2], [2,2,0], [1,2,0], [3,2,0]]),
}

# Parametri NewsVendor

parameters_NewsVendor = {
    'cost' : 3,
    'selling_price' : 10,
}


