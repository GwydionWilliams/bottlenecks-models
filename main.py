import numpy as np
from scipy.optimize import minimize, Bounds
import pandas as pd
import agentClasses
from simulation import Simulation

# -----------------------------------------------------------------------------
# 1. INITIALISE PARAMETERS ----------------------------------------------------
#    i.   SIMULATION MODE
simParams = {
    "mode": "simulate",
    "numTrials": 200,
    "alphas": [1.0],  # np.arange(.2, 1.2, .2),
    "betas": np.arange(.2, 1.2, .2),
    "phis": [0],
    "etas": [1],
    "numReps": 20,
    "writeData": False,
    "dataPath": "./data/simulatedBehaviour.csv"
}

models, m_i = [
    "flat-history",             # 0
    "hierarchical",             # 1
    "structured-hierarchical",  # 2.
    "abstract-hierarchical"     # 3
], 1

fixedAgentParams = {
    "gamma": .5,
    "talkative": True
}

envParams = {
    "regimes": ["REP", "ALT"]
}

# -----------------------------------------------------------------------------
# 2. RUN SIMULATION -----------------------------------------------------------
sim = Simulation(simParams, envParams)

if sim.mode == "simulate":
    if sim.writeData:
        sim.initDataFile()

    for model in models:
        agentParams = agentClasses.params[model]
        agentParams.update(fixedAgentParams)
        sim.simulate(agentParams)

elif sim.mode == "fit":
    d = pd.read_csv("./data/empiricalData.csv")

    d = d.loc[
        :, ["subjNum", "policySet", "episode", "domain", "policy", "path",
            "statesVisited", "actions", "steps"]
    ]

    if sim.writeData:
        f = open("data/modelFits-ideal.csv", "w")
        f.write("model,subj,a_0,b_0,p_0,h_0,a_opt,b_opt,p_opt,h_opt,nll\n")
        f.close()

    for model in models:
        agentParams = agentClasses.params[model]
        agentParams.update(fixedAgentParams)

        if "hierarchical" in model:
            simParams["etas"] = [0.25, 0.75]
            simParams["phis"] = [0.01, 0.1]

            if "sequential" in agentParams["selectionStrategy"]:
                bounds = Bounds([0.01, 0.01, 0.01, 0.75], [1, 5, .99, 1])
            else:
                simParams["etas"] = [0]
                bounds = Bounds([0.01, 0.01, 0.01], [1, 5, .99])
        else:
            simParams["etas"] = [0]
            simParams["phis"] = [0]

            bounds = Bounds([0.01, 0.01], [1, 5])

        for subj in d.subjNum.unique():
            d_s = d[d.subjNum == subj].reset_index()

            for a in sim.alphas:
                for b in sim.betas:
                    for p in sim.phis:
                        for h in sim.etas:
                            print("\n", subj, a, b, p, h)
                            x0 = np.array([a, b])

                            if "hierarchical" in model:
                                x0 = np.append(x0, p)
                                agentParams["eta"] = h

                                if "sequential" in agentParams["selectionStrategy"]:
                                    x0 = np.append(x0, h)
                                else:
                                    agentParams["eta"] = h

                            else:
                                agentParams["phi"] = p
                                agentParams["eta"] = h

                            sol = minimize(
                                sim.fit, x0, args=(d_s, agentParams),
                                bounds=bounds
                            )

                            while len(x0) < 4:
                                x0 = np.append(x0, "NA")
                                sol.x = np.append(sol.x, "NA")

                            rowData = [model, subj]
                            rowData = np.append(rowData, x0)
                            rowData = np.append(rowData, sol.x)
                            rowData = np.append(rowData, sol.fun)

                            row = ",".join([str(item) for item in rowData]) \
                                + "\n"

                            f = open("data/modelFits-ideal.csv", "a")
                            f.write(row)
                            f.close()

                            print(sol)

        # print(sim.fit([.1, .5], d, agentParams))
