import gurobipy as gp
from gurobipy import GRB

w = [0.8,0.6,0.7,0.4,0.5,0.3,0.2]   # weights
K = 3
n = len(w)

m = gp.Model("maxsat_lp")
m.setParam("OutputFlag", 0)

# x[i] = truth value of Boolean variable
x = m.addVars(n, lb=0, ub=1, name="x")

# z[i] = satisfaction of (x_i)
z = m.addVars(n, lb=0, ub=1, name="z")

# z_i <= x_i
for i in range(n):
    m.addConstr(z[i] <= x[i], name=f"link_{i}")

# Capacity constraint: at most K chosen
m.addConstr(gp.quicksum(x[i] for i in range(n)) <= K, name="capacity")

# Objective: maximize sum_i w_i * z_i
m.setObjective(gp.quicksum(w[i] * z[i] for i in range(n)), GRB.MAXIMIZE)

m.optimize()

for i in range(n):
    print(f"{i}: x={x[i].X:.3f},  z={z[i].X:.3f}")