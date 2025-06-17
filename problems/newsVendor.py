import gurobipy as gp
from gurobipy import GRB
from data import *

def newsVendor(demand, prob, parameters):

    n_scenarios = len(demand)
    selling_price = parameters.get('selling_price')
    cost = parameters.get('cost')
    exp_val = 0

    # Model
    m = gp.Model("newsvendor")
    m.setParam('OutputFlag', 0)

    n_neswpaper = m.addVar(vtype=GRB.INTEGER, lb=0, name="X") # Number of bought newspaper
    y = m.addVars(n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y") # Numeber of sold newspaper

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

    return ottimo, m.ObjVal

# Function implemented to study the out-of-sample stability
def compute_objective_NewsVendor(solution, demand, parameters):

    selling_price = parameters.get('selling_price')
    cost = parameters.get('cost')

    sold = min(solution, demand)

    return selling_price * sold - cost * solution


