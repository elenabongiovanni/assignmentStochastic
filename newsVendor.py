import gurobipy as gp
from gurobipy import GRB

cost = 4
selling_price = 10 
pi = [0.4, 0.3, 0.2, 0.1]
vals = [0,1,2,3]
n_scenarios = len(vals)
scenarios = range(n_scenarios)

# Model
m = gp.Model("newsvendor")

n_neswpaper = m.addVar(vtype=GRB.INTEGER, lb=0, name="X")
y = m.addVars(n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y")

exp_val = sum(pi[s] * y[s] for s in scenarios) 
m.setObjective(
    selling_price * exp_val - cost * n_neswpaper,
    GRB.MAXIMIZE
)

for s in scenarios:
    print(s)
    m.addConstr(
        y[s] <= n_neswpaper
    )
    m.addConstr(
        y[s] <= vals[s]
    )

# Save model
# m.write("newsvendor.lp")

# Solve
m.optimize()
ottimo = m.objVal
print(ottimo)
#calcolo funzione obbiettivo per tutti gli scenari con x=ottimo

#print(f"n_neswpaper: {n_neswpaper.X}")