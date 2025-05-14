import gurobipy as gp
from gurobipy import GRB
from instances import *
from data import *


def newsVendor(demand, prob, parameters):

    n_scenarios = len(demand)
    selling_price = parameters.get('selling_price')
    cost = parameters.get('cost')
    exp_val = 0

    scenarios = demand

    # Model
    m = gp.Model("newsvendor")
    m.setParam('OutputFlag', 0)

    n_neswpaper = m.addVar(vtype=GRB.INTEGER, lb=0, name="X") #number of bought newspaper
    y = m.addVars(n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y") #numeber of sold newspaper

    exp_val = sum(prob[i] * y[i] for i in range(n_scenarios)) 
    m.setObjective(
        selling_price * exp_val - cost * n_neswpaper,
        GRB.MAXIMIZE
    )

    for i in range(n_scenarios):
        m.addConstr(
            y[i] <= n_neswpaper
        )
        m.addConstr(
            y[i] <= demand[i]
        )

    # Solve
    m.optimize()
    ottimo = n_neswpaper.X
    print(ottimo)

    return ottimo, m.ObjVal

def compute_objective_NewsVendor(solution, demand, parameters):
    """
    Calcola il valore obiettivo (profitto) di una soluzione fissata nel problema del Newsvendor.
    """
    selling_price = parameters.get('selling_price')
    cost = parameters.get('cost')

    sold = min(solution, demand)
    return selling_price * sold - cost * solution


