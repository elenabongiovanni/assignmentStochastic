import gurobipy as gp
from gurobipy import GRB
from instances import *
from data import *

def assembleToOrder(demand, prob, parameters):
 
    exp_val = 0
    n_scenarios = len(demand)
    n_components = parameters.get('n_components')
    n_items = parameters.get('n_items')
    cost = parameters.get('costs')
    selling_price = parameters.get('selling_prices')
    n_machines = parameters.get('n_machines')
    l = parameters.get('machine_capacities')
    g = parameters.get('gozinto_factor')
    t = parameters.get('time_required')

    scenarios = demand

    # Model
    m = gp.Model("assembleToOrder")
    m.setParam('OutputFlag', 0)

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
                sum( g[i,j]*y[j,s] for j in range(n_items)) <= x[i] # Capacity items production
            )
    for k in range(n_machines):
        m.addConstr(
            sum( x[i]*t[i,k] for i in range(n_components)) <= l[k] # Capacity production on machines
        )

    # Solve
    m.optimize()
    ottimo = [x[i].X for i in range(n_components)]
    print(ottimo)

    print(f"Quantity of each component: {ottimo}")
    print(m.ObjVal)

    return ottimo, m.ObjVal
