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


# da riguardare copiata da chat!!!
def compute_objective_ATO(solution, demand, parameters):
    """
    Valuta il valore della soluzione fissa `solution` (componenti ordinate)
    su una nuova realizzazione di domanda `demand`.

    solution: lista con quantità ordinate di ogni componente (x)
    demand: array 2D [n_items x n_scenarios]
    """
    n_scenarios = demand.shape[1]
    n_components = parameters.get('n_components')
    n_items = parameters.get('n_items')
    selling_price = parameters.get('selling_prices')
    cost = parameters.get('costs')
    g = parameters.get('gozinto_factor')  # matrice [n_components x n_items]

    total_value = 0

    for s in range(n_scenarios):
        revenue = 0
        # Calcola i prodotti effettivamente assemblabili per lo scenario s
        remaining_components = solution.copy()  # componenti disponibili

        y = [0] * n_items  # prodotti venduti nello scenario s

        # Strategia greedy: prova a soddisfare la domanda usando i componenti disponibili
        for j in range(n_items):
            max_demand = demand[j, s]
            max_producible = min(
                max_demand,
                min([
                    remaining_components[i] // g[i][j] if g[i][j] > 0 else float('inf')
                    for i in range(n_components)
                ])
            )
            y[j] = max_producible
            for i in range(n_components):
                remaining_components[i] -= g[i][j] * y[j]

            revenue += selling_price[j] * y[j]

        cost_components = sum(cost[i] * solution[i] for i in range(n_components))
        value = revenue - cost_components
        total_value += value / n_scenarios

    return total_value
