import gurobipy as gp
from gurobipy import GRB
import numpy as np
from instances import *
from data import *

def plantLocation(demand, prob, transportation_cost, fixed_cost):

    n_production_nodes = 5
    n_demand_nodes = 7
    capacity_level = np.ones(n_demand_nodes)
    n_scenarios = 4
    penalty_coefficient = 3*np.ones(n_demand_nodes)
    scenarios = []
    for i in range(n_scenarios):
        s = Scenario(demand[i], prob[i])
        scenarios.append(s)

    # Model
    m = gp.Model("assembleToOrder")

    # Decision variables
    y = m.addVars(n_production_nodes, vtype=GRB.BINARY, name="Y")
    z = m.addVars(n_demand_nodes, n_scenarios, vtype=GRB.CONTINUOUS, name="Z")
    x = m.addVars(n_production_nodes, n_demand_nodes, n_scenarios, vtype=GRB.CONTINUOUS, lb=0, name="X")

    # Objective Function
    opening_arc = sum(fixed_cost[i] * y[i] for i in range(n_production_nodes))
    for s in range(n_scenarios):
        exp_val = exp_val + prob[s] * sum( transportation_cost[i,j] * x[i,j,s] for i in range(n_production_nodes) for j in range(n_demand_nodes))
        exp_penalty_val = exp_penalty_val + prob[s] * sum( penalty_coefficient[j] * z[j,s] for j in range(n_demand_nodes))
    m.setObjective(
        exp_val + opening_arc + exp_penalty_val,
        GRB.MAXIMIZE
    )

    # Constraints
    for s in range(n_scenarios):
        for j in range(n_demand_nodes):
            sum( x[i,j,s] for i in range(n_production_nodes) ) + z[j,s] == demand[j,s]

        for i in range(n_production_nodes):
            sum( x[i,j,s] for j in range(n_demand_nodes) ) <= capacity_level[i]*y[i]

    # Solve
    m.optimize()
    ottimo = y.Y

    print(ottimo)
    print(m.ObjVal)

    return ottimo, m.ObjVal
