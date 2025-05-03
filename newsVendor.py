import gurobipy as gp
from gurobipy import GRB
from instances import *
from data import *

def newsVendor(demand, prob, selling_price, cost):
    n_scenarios = len(demand)
    scenarios = []
    for i in range(n_scenarios):
        s = Scenario(demand[i], prob[i])
        scenarios.append(s)

    # Model
    m = gp.Model("newsvendor")

    n_neswpaper = m.addVar(vtype=GRB.INTEGER, lb=0, name="X") #number of bought newspaper
    y = m.addVars(n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y") #numeber of sold newspaper

    exp_val = sum(prob[i] * y[i] for i in range(n_scenarios)) 
    m.setObjective(
        selling_price * exp_val - cost * n_neswpaper,
        GRB.MAXIMIZE
    )

    for i in range(n_scenarios):
        #print(i)
        m.addConstr(
            y[i] <= n_neswpaper
        )
        m.addConstr(
            y[i] <= demand[i]
        )

    # Save model
    # m.write("newsvendor.lp")

    # Solve
    m.optimize()
    ottimo = n_neswpaper.X
    print(ottimo)

    #calcolo funzione obbiettivo per tutti gli scenari con x=ottimo
    """for i in range(n_scenarios):
        obj_value_s = selling_price * min(ottimo,demand[i]) - cost * ottimo
        scenarios[i].add_gain(obj_value_s)
        print(f"Scenario {i}: Valore della funzione obiettivo = {obj_value_s}")"""

    return ottimo, m.ObjVal
