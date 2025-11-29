import gurobipy as gp
from gurobipy import GRB

try:
    m = gp.Model("chocolate_model")

    # Variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")

    # Objective
    m.setObjective(-2000*x - 3000*y, GRB.MINIMIZE)

    # Constraints
    m.addConstr(0.5 * x + 0.2 * y <= 2, name="c0")
    m.addConstr(10 * x + 30 * y <= 70, name="c1")
    m.addConstr(x >= 0, name="c2")
    m.addConstr(y >= 0, name="c3")

    # Optimize
    m.optimize()
except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")