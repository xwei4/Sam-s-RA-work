import gurobipy as gp
from gurobipy import GRB
import random

n = 7   

P_list = [
    [0, 1],    # clause 0: x0 OR x1
    [2],       # clause 1: x2
    [1, 3],    # clause 2: x1 OR x3
    [4],       # clause 3: x4
    [0, 5],    # clause 4: x0 OR x5
    [6],       # clause 5: x6
    [2, 4, 5, 6]  # clause 6: x2 OR x4 OR x5 OR x6
]

N_list = [
    [],       # clause 0: no negated literals
    [1],      # clause 1: (x2 OR not x1)
    [],       # clause 2:
    [2],      # clause 3: (x4 OR not x2)
    [3],      # clause 4: (x0 OR x5 OR not x3)
    [0],      # clause 5: (x6 OR not x0)
    []        # clause 6: 
]

# weights 
w = [random.random() for _ in range(n)]

model = gp.Model("maxsat_lp_relaxation")

# y_i in [0,1]
y = model.addVars(n, lb=0.0, ub=1.0, name="y")

# z_j in [0,1]
z = model.addVars(n, lb=0.0, ub=1.0, name="z")

# constraints:
for j in range(n):
    pos = gp.quicksum(y[i] for i in P_list[j]) if P_list[j] else 0
    neg = gp.quicksum((1 - y[i]) for i in N_list[j]) if N_list[j] else 0
    model.addConstr(pos + neg >= z[j], name=f"C{j}")


# Objective
model.setObjective(gp.quicksum(w[j] * z[j] for j in range(n)), GRB.MAXIMIZE)

model.optimize()

print("Objective:", model.ObjVal)
print("Variable")
for i in range(n):
    print(f" y[{i}] = {y[i].X}")
print("Clause satisfaction (z_j):")
for j in range(n):
    print(f" z[{j}] = {z[j].X}   (weight {w[j]})")
