import gurobipy as gp
import sys
from gurobipy import Model, GRB, quicksum

def read_maxsat_instance(path):
    with open(path, "r") as f:
        first_line = f.readline()
        n, m = map(int, first_line.split())
        clauses = []

        for _ in range(m):
            line = f.readline()
            if line == "":
                raise ValueError("Not enough clause lines in the input file.")

            parts = line.split()

            weight = float(parts[0])
            lits = list(map(int, parts[1:]))

            P = []
            N = []
            for lit in lits:
                if lit > 0:
                    P.append(lit)
                else:
                    N.append(-lit)

            clauses.append({"weight": weight, "P": P, "N": N})

    return n, m, clauses


def build_and_solve_lp(n, m, clauses, verbose=True):
    model = Model("maxsat_lp")

    # y_i variables: fractional truth values for each variable
    y = {}
    for i in range(1, n + 1):
        y[i] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y_{i}")

    # z_j variables: fractional satisfaction for each clause
    z = {}
    for j in range(1, m + 1):
        z[j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{j}")

    model.update()

    # Add constraints for each clause
    for j in range(1, m + 1):
        clause = clauses[j - 1]
        P_j = clause["P"]
        N_j = clause["N"]

        lhs = quicksum(y[i] for i in P_j) + quicksum(1 - y[i] for i in N_j)
        model.addConstr(lhs >= z[j], name=f"clause_{j}")

    # Objective: maximize sum_j w_j * z_j
    obj = quicksum(clauses[j - 1]["weight"] * z[j] for j in range(1, m + 1))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Solve the LP
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Model did not solve to optimality. Status: {model.status}")
        return None

    print(f"Optimal LP objective value (sum w_j z_j): {model.objVal}")

    print("\nFractional variable values y_i (LP solution):")
    for i in range(1, n + 1):
        print(f"  y_{i} = {y[i].X:.4f}")

    print("\nFractional clause values z_j:")
    for j in range(1, m + 1):
        print(f"  z_{j} = {z[j].X:.4f}")

    return model, y, z


def main():
    if len(sys.argv) != 2:
        print("Usage: python maxsat_lp.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    n, m, clauses = read_maxsat_instance(input_file)
    print(f"Read MAX-SAT instance with {n} variables and {m} clauses from {input_file}.")
    build_and_solve_lp(n, m, clauses, verbose=True)


if __name__ == "__main__":
    main()