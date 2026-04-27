import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_model(p, thetaL, thetaH, eps=1e-5, time_limit=1.0):
    deltaTheta = thetaH - thetaL

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.NonConvex = 2

    m.Params.TimeLimit = time_limit

    qHc = m.addVar(lb=eps, name="qHc")
    qHStar = m.addVar(lb=eps, name="qHstar")
    qLStar = m.addVar(lb=eps, name="qLstar")

    VqHc = m.addVar(lb=0, name="VqHc")
    VqHStar = m.addVar(lb=0, name="VqHstar")
    VqLStar = m.addVar(lb=0, name="VqLstar")

    z = m.addVar(lb=0, ub=1, name="dynamic_valuation")

    m.addConstr(qHStar >= qHc + eps)
    m.addConstr(qLStar >= qHStar + eps)

    m.addConstr(VqHStar >= VqHc + eps)
    m.addConstr(VqLStar >= VqHStar + eps)

    S = (p * deltaTheta) / (1 - p) + thetaH

    m.addConstr(VqHc >= S)
    m.addConstr(VqHStar - VqHc <= S * qHc * (qHStar - qHc))
    m.addConstr(VqHStar - VqHc >= thetaH * (qHStar - qHc))
    m.addConstr(VqLStar - VqHStar <= thetaH * (qLStar - qHStar))
    m.addConstr(VqLStar - VqHStar >= thetaL * (qLStar - qHStar))

    m.addConstr(z >= VqHStar - thetaH * qHStar)
    m.addConstr(z >= p * (VqLStar - thetaL * qLStar))

    m.addConstr(
        p * (VqLStar - thetaL * qLStar - deltaTheta * qHc)
        + (1 - p) * (VqHc - thetaH * qHc)
        == 1
    )

    m.setObjective(z, GRB.MINIMIZE)

    m.optimize()

    if m.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        return None

    return {
        "p": p,
        "thetaL": thetaL,
        "thetaH": thetaH,
        "qHc": qHc.X,
        "qHStar": qHStar.X,
        "qLStar": qLStar.X,
        "VqHc": VqHc.X,
        "VqHStar": VqHStar.X,
        "VqLStar": VqLStar.X,
        "z": z.X,
        "objective_z": m.objVal,
    }


def search():
    p_center = 0.5
    thetaH_center = 5
    ratio_center = 0.5

    p_width = 0.4
    thetaH_width = 4
    ratio_width = 0.4

    rounds = 3
    best = None

    eps = 1e-6

    for r in range(rounds):
        print(f"\n  ROUND {r+1}  ")

        p_vals = np.linspace(
            max(eps, p_center - p_width),
            min(1 - eps, p_center + p_width),
            5
        )

        thetaH_vals = np.linspace(
            max(0.5, thetaH_center - thetaH_width),
            thetaH_center + thetaH_width,
            5
        )

        ratio_vals = np.linspace(
            max(eps, ratio_center - ratio_width),
            min(1 - eps, ratio_center + ratio_width),
            5
        )

        best_obj = float("inf")
        best = None

        for p in p_vals:
            for thetaH in thetaH_vals:
                for ratio in ratio_vals:

                    thetaL = ratio * thetaH

                    if thetaL >= thetaH:
                        continue

                    if thetaH < 0.5:
                        continue

                    result = solve_model(p, thetaL, thetaH)

                    if result is None:
                        continue

                    if result["objective_z"] < best_obj:
                        best_obj = result["objective_z"]
                        best = result

        if best is None:
            print("No feasible solution found in this round.")
            break

        print("\nBest so far:")
        for k, v in best.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        # zoom in
        p_center = best["p"]
        thetaH_center = best["thetaH"]
        ratio_center = best["thetaL"] / best["thetaH"]

        p_width *= 0.4
        thetaH_width *= 0.4
        ratio_width *= 0.4

    return best


best_solution = search()

print("\n FINAL SOLUTION ")
if best_solution:
    for k, v in best_solution.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
else:
    print("No solution found.")