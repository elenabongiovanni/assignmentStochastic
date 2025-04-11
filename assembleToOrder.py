import gurobipy as gp
from gurobipy import GRB
from instances import *
from data import *

def assembleToOrder(demand, prob, selling_price, cost):

    # settare domanda e probabilità !!!!!!!!!!!!!

    # Data set initialization
    # CAMBIARE !!!!!!
    n_components = 5
    n_items = 2
    n_machines = 2
    n_scenarios = len(demand)
    g = ones(n_components, n_items)
    t = ones(n_components, n_machines)
    l = ones(n_machines)

    scenarios = []
    for i in range(n_scenarios):
        s = Scenario(demand[i], prob[i])
        scenarios.append(s)

    # Model
    m = gp.Model("assembleToOrder")

    x = m.addVars(n_components, vtype=GRB.INTEGER, lb=0, name="X") #quantity of each component
    y = m.addVars(n_items, n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y") #quantity of each item

    # Objective Function
    sum_costs_components = sum(cost[i] * x[i] for i in range(n_components))
    for s in range(n_scenarios):
        exp_val = exp_val + prob[s] * sum( selling_price[j] * y[j,s] for j in range(n_items))
    m.setObjective(
        exp_val - sum_costs_components,
        GRB.MAXIMIZE
    )

    # Constraints
    for s in range(n_scenarios):
        for j in range(n_items):
            m.addConstr(
                y[j,s] <= demand[j,s] # Demand satisfaction
            )
        for i in range(n_components):
            m.addConstr(
                sum( g[i,j]*y[j,s] for j in range(n_item)) <= x[i] # Capacity items production
            )
    for k in range(n_machines):
        m.addConstr(
            sum( t[i,k] * x[i] for i in range(n_components)) <= l[k] # Capacity production on machines
        )

    # Solve
    m.optimize()
    ottimo = x.X
    print(ottimo)

    print(f"Quantity of each component: {x.X}")
    print(m.ObjVal)

    return ottimo, m.ObjVal
